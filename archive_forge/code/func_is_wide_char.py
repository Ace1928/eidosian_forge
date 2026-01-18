from __future__ import annotations
import re
import typing
import warnings
import wcwidth
def is_wide_char(text: str | bytes, offs: int) -> bool:
    """
    Test if the character at offs within text is wide.

    text may be unicode or a byte string in the target _byte_encoding
    """
    if isinstance(text, str):
        return get_char_width(text[offs]) == 2
    if not isinstance(text, bytes):
        raise TypeError(text)
    if _byte_encoding == 'utf8':
        o, _n = decode_one(text, offs)
        return get_width(o) == 2
    if _byte_encoding == 'wide':
        return within_double_byte(text, offs, offs) == 1
    return False