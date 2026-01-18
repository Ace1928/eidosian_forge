from typing import Callable, Dict, Iterable, Union
from Bio.Align import MultipleSeqAlignment
from Bio.File import as_handle
from Bio.SeqIO import AbiIO
from Bio.SeqIO import AceIO
from Bio.SeqIO import FastaIO
from Bio.SeqIO import GckIO
from Bio.SeqIO import IgIO  # IntelliGenetics or MASE format
from Bio.SeqIO import InsdcIO  # EMBL and GenBank
from Bio.SeqIO import NibIO
from Bio.SeqIO import PdbIO
from Bio.SeqIO import PhdIO
from Bio.SeqIO import PirIO
from Bio.SeqIO import QualityIO  # FastQ and qual files
from Bio.SeqIO import SeqXmlIO
from Bio.SeqIO import SffIO
from Bio.SeqIO import SnapGeneIO
from Bio.SeqIO import SwissIO
from Bio.SeqIO import TabIO
from Bio.SeqIO import TwoBitIO
from Bio.SeqIO import UniprotIO
from Bio.SeqIO import XdnaIO
from Bio.SeqRecord import SeqRecord
from .Interfaces import _TextIOSource
def index_db(index_filename, filenames=None, format=None, alphabet=None, key_function=None):
    """Index several sequence files and return a dictionary like object.

    The index is stored in an SQLite database rather than in memory (as in the
    Bio.SeqIO.index(...) function).

    Arguments:
     - index_filename - Where to store the SQLite index
     - filenames - list of strings specifying file(s) to be indexed, or when
       indexing a single file this can be given as a string.
       (optional if reloading an existing index, but must match)
     - format   - lower case string describing the file format
       (optional if reloading an existing index, but must match)
     - alphabet - no longer used, leave as None.
     - key_function - Optional callback function which when given a
       SeqRecord identifier string should return a unique
       key for the dictionary.

    This indexing function will return a dictionary like object, giving the
    SeqRecord objects as values:

    >>> from Bio import SeqIO
    >>> files = ["GenBank/NC_000932.faa", "GenBank/NC_005816.faa"]
    >>> def get_gi(name):
    ...     parts = name.split("|")
    ...     i = parts.index("gi")
    ...     assert i != -1
    ...     return parts[i+1]
    >>> idx_name = ":memory:" #use an in memory SQLite DB for this test
    >>> records = SeqIO.index_db(idx_name, files, "fasta", key_function=get_gi)
    >>> len(records)
    95
    >>> records["7525076"].description
    'gi|7525076|ref|NP_051101.1| Ycf2 [Arabidopsis thaliana]'
    >>> records["45478717"].description
    'gi|45478717|ref|NP_995572.1| pesticin [Yersinia pestis biovar Microtus str. 91001]'
    >>> records.close()

    In this example the two files contain 85 and 10 records respectively.

    BGZF compressed files are supported, and detected automatically. Ordinary
    GZIP compressed files are not supported.

    See Also: Bio.SeqIO.index() and Bio.SeqIO.to_dict(), and the Python module
    glob which is useful for building lists of files.

    """
    from os import fspath

    def is_pathlike(obj):
        """Test if the given object can be accepted as a path."""
        try:
            fspath(obj)
            return True
        except TypeError:
            return False
    if not is_pathlike(index_filename):
        raise TypeError('Need a string or path-like object for filename (not a handle)')
    if is_pathlike(filenames):
        filenames = [filenames]
    if filenames is not None and (not isinstance(filenames, list)):
        raise TypeError('Need a list of filenames (as strings or path-like objects), or one filename')
    if format is not None and (not isinstance(format, str)):
        raise TypeError('Need a string for the file format (lower case)')
    if format and (not format.islower()):
        raise ValueError(f"Format string '{format}' should be lower case")
    if alphabet is not None:
        raise ValueError('The alphabet argument is no longer supported')
    from ._index import _FormatToRandomAccess
    from Bio.File import _SQLiteManySeqFilesDict
    repr = 'SeqIO.index_db(%r, filenames=%r, format=%r, key_function=%r)' % (index_filename, filenames, format, key_function)

    def proxy_factory(format, filename=None):
        """Given a filename returns proxy object, else boolean if format OK."""
        if filename:
            return _FormatToRandomAccess[format](filename, format)
        else:
            return format in _FormatToRandomAccess
    return _SQLiteManySeqFilesDict(index_filename, filenames, proxy_factory, format, key_function, repr)