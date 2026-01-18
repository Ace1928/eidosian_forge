import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def pretty_use_unicode(flag=None):
    """Set whether pretty-printer should use unicode by default"""
    global _use_unicode
    global unicode_warnings
    if flag is None:
        return _use_unicode
    if flag and unicode_warnings:
        warnings.warn(unicode_warnings)
        unicode_warnings = ''
    use_unicode_prev = _use_unicode
    _use_unicode = flag
    return use_unicode_prev