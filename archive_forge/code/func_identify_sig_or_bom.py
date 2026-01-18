import importlib
import logging
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder  # type: ignore
from .constant import (
def identify_sig_or_bom(sequence: bytes) -> Tuple[Optional[str], bytes]:
    """
    Identify and extract SIG/BOM in given sequence.
    """
    for iana_encoding in ENCODING_MARKS:
        marks = ENCODING_MARKS[iana_encoding]
        if isinstance(marks, bytes):
            marks = [marks]
        for mark in marks:
            if sequence.startswith(mark):
                return (iana_encoding, mark)
    return (None, b'')