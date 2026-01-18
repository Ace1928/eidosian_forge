from __future__ import annotations
import re
import typing
import warnings
import wcwidth
def decode_one_uni(text: str, i: int) -> tuple[int, int]:
    """
    decode_one implementation for unicode strings
    """
    return (ord(text[i]), i + 1)