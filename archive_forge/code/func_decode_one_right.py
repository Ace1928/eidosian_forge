from __future__ import annotations
import re
import typing
import warnings
import wcwidth
def decode_one_right(text: bytes, pos: int) -> tuple[int, int] | None:
    """
    Return (ordinal at pos, next position) for UTF-8 encoded text.
    pos is assumed to be on the trailing byte of a utf-8 sequence.
    """
    if not isinstance(text, bytes):
        raise TypeError(text)
    error = (ord('?'), pos - 1)
    p = pos
    while p >= 0:
        if text[p] & 192 != 128:
            o, _next_pos = decode_one(text, p)
            return (o, p - 1)
        p -= 1
        if p == p - 4:
            return error
    return None