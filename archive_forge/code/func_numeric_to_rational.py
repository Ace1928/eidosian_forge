import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def numeric_to_rational(numeric):
    """Converts a numeric string to a rational string, if possible."""
    if numeric[:1] == '-':
        sign, numeric = (numeric[0], numeric[1:])
    else:
        sign = ''
    parts = numeric.split('/')
    if len(parts) == 2:
        num, den = float_to_rational(float(parts[0]) / float(parts[1]))
    elif len(parts) == 1:
        num, den = float_to_rational(float(parts[0]))
    else:
        raise ValueError()
    result = '{}{}/{}'.format(sign, num, den)
    if result.endswith('/1'):
        return result[:-2]
    return result