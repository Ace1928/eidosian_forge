import importlib
import logging
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder  # type: ignore
from .constant import (
@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_latin(character: str) -> bool:
    try:
        description = unicodedata.name(character)
    except ValueError:
        return False
    return 'LATIN' in description