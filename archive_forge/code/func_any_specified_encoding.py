import importlib
import logging
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder  # type: ignore
from .constant import (
def any_specified_encoding(sequence: bytes, search_zone: int=4096) -> Optional[str]:
    """
    Extract using ASCII-only decoder any specified encoding in the first n-bytes.
    """
    if not isinstance(sequence, bytes):
        raise TypeError
    seq_len = len(sequence)
    results = findall(RE_POSSIBLE_ENCODING_INDICATION, sequence[:min(seq_len, search_zone)].decode('ascii', errors='ignore'))
    if len(results) == 0:
        return None
    for specified_encoding in results:
        specified_encoding = specified_encoding.lower().replace('-', '_')
        for encoding_alias, encoding_iana in aliases.items():
            if encoding_alias == specified_encoding:
                return encoding_iana
            if encoding_iana == specified_encoding:
                return encoding_iana
    return None