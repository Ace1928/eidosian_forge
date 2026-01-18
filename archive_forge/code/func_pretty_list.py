import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def pretty_list(l, mapping):
    result = []
    for s in l:
        pretty = mapping.get(s)
        if pretty is None:
            try:
                pretty = ''.join([mapping[c] for c in s])
            except (TypeError, KeyError):
                return None
        result.append(pretty)
    return result