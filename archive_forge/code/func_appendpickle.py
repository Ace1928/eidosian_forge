from __future__ import absolute_import, print_function, division
from petl.compat import pickle, next
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def appendpickle(table, source=None, protocol=-1, write_header=False):
    """
    Append data to an existing pickle file. I.e.,
    as :func:`petl.io.pickle.topickle` but the file is opened in append mode.

    Note that no attempt is made to check that the fields or row lengths are
    consistent with the existing data, the data rows from the table are simply
    appended to the file.

    """
    _writepickle(table, source=source, mode='ab', protocol=protocol, write_header=write_header)