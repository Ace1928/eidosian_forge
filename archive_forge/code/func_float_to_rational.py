import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def float_to_rational(flt):
    """Converts a float to a rational pair."""
    int_part = int(flt)
    error = flt - int_part
    if abs(error) < 0.0001:
        return (int_part, 1)
    den, num = float_to_rational(1.0 / error)
    return (int_part * den + num, den)