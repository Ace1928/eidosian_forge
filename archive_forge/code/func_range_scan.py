import importlib
import logging
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder  # type: ignore
from .constant import (
def range_scan(decoded_sequence: str) -> List[str]:
    ranges = set()
    for character in decoded_sequence:
        character_range = unicode_range(character)
        if character_range is None:
            continue
        ranges.add(character_range)
    return list(ranges)