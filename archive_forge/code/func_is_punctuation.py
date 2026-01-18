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
def is_punctuation(character: str) -> bool:
    character_category = unicodedata.category(character)
    if 'P' in character_category:
        return True
    character_range = unicode_range(character)
    if character_range is None:
        return False
    return 'Punctuation' in character_range