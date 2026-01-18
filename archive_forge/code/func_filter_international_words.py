import logging
import re
from typing import Optional, Union
from .enums import LanguageFilter, ProbingState
@staticmethod
def filter_international_words(buf: Union[bytes, bytearray]) -> bytearray:
    """
        We define three types of bytes:
        alphabet: english alphabets [a-zA-Z]
        international: international characters [\x80-ÿ]
        marker: everything else [^a-zA-Z\x80-ÿ]
        The input buffer can be thought to contain a series of words delimited
        by markers. This function works to filter all words that contain at
        least one international character. All contiguous sequences of markers
        are replaced by a single space ascii character.
        This filter applies to all scripts which do not use English characters.
        """
    filtered = bytearray()
    words = INTERNATIONAL_WORDS_PATTERN.findall(buf)
    for word in words:
        filtered.extend(word[:-1])
        last_char = word[-1:]
        if not last_char.isalpha() and last_char < b'\x80':
            last_char = b' '
        filtered.extend(last_char)
    return filtered