import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def pretty_try_use_unicode():
    """See if unicode output is available and leverage it if possible"""
    encoding = getattr(sys.stdout, 'encoding', None)
    if encoding is None:
        return
    symbols = []
    symbols += greek_unicode.values()
    symbols += atoms_table.values()
    for s in symbols:
        if s is None:
            return
        try:
            s.encode(encoding)
        except UnicodeEncodeError:
            return
    pretty_use_unicode(True)