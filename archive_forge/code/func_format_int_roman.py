import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def format_int_roman(value: int) -> str:
    """Format a number as lowercase Roman numerals."""
    assert 0 < value < 4000
    result: List[str] = []
    index = 0
    while value != 0:
        value, remainder = divmod(value, 10)
        if remainder == 9:
            result.insert(0, ROMAN_ONES[index])
            result.insert(1, ROMAN_ONES[index + 1])
        elif remainder == 4:
            result.insert(0, ROMAN_ONES[index])
            result.insert(1, ROMAN_FIVES[index])
        else:
            over_five = remainder >= 5
            if over_five:
                result.insert(0, ROMAN_FIVES[index])
                remainder -= 5
            result.insert(1 if over_five else 0, ROMAN_ONES[index] * remainder)
        index += 1
    return ''.join(result)