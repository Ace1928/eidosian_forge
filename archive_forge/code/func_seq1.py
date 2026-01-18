import re
import warnings
from math import pi, sin, cos, log, exp
from Bio.Seq import Seq, complement, complement_rna, translate
from Bio.Data import IUPACData
from Bio.Data.CodonTable import standard_dna_table
from Bio import BiopythonDeprecationWarning
def seq1(seq, custom_map=None, undef_code='X'):
    """Convert protein sequence from three-letter to one-letter code.

    The single required input argument 'seq' should be a protein sequence
    using three-letter codes, either as a Python string or as a Seq or
    MutableSeq object.

    This function returns the amino acid sequence as a string using the one
    letter amino acid codes. Output follows the IUPAC standard (including
    ambiguous characters "B" for "Asx", "J" for "Xle", "X" for "Xaa", "U" for
    "Sel", and "O" for "Pyl") plus "*" for a terminator given the "Ter" code.
    Any unknown character (including possible gap characters), is changed
    into '-' by default.

    e.g.

    >>> from Bio.SeqUtils import seq1
    >>> seq1("MetAlaIleValMetGlyArgTrpLysGlyAlaArgTer")
    'MAIVMGRWKGAR*'

    The input is case insensitive, e.g.

    >>> from Bio.SeqUtils import seq1
    >>> seq1("METalaIlEValMetGLYArgtRplysGlyAlaARGTer")
    'MAIVMGRWKGAR*'

    You can set a custom translation of the codon termination code using the
    dictionary "custom_map" argument (defaulting to {'Ter': '*'}), e.g.

    >>> seq1("MetAlaIleValMetGlyArgTrpLysGlyAla***", custom_map={"***": "*"})
    'MAIVMGRWKGA*'

    You can also set a custom translation for non-amino acid characters, such
    as '-', using the "undef_code" argument, e.g.

    >>> seq1("MetAlaIleValMetGlyArgTrpLysGlyAla------ArgTer", undef_code='?')
    'MAIVMGRWKGA??R*'

    If not given, "undef_code" defaults to "X", e.g.

    >>> seq1("MetAlaIleValMetGlyArgTrpLysGlyAla------ArgTer")
    'MAIVMGRWKGAXXR*'

    """
    if custom_map is None:
        custom_map = {'Ter': '*'}
    onecode = {k.upper(): v for k, v in IUPACData.protein_letters_3to1_extended.items()}
    onecode.update(((k.upper(), v) for k, v in custom_map.items()))
    seqlist = [seq[3 * i:3 * (i + 1)] for i in range(len(seq) // 3)]
    return ''.join((onecode.get(aa.upper(), undef_code) for aa in seqlist))