import importlib
import logging
import unicodedata
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import Generator, List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder
from .constant import (
@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_arabic(character: str) -> bool:
    try:
        character_name = unicodedata.name(character)
    except ValueError:
        return False
    return 'ARABIC' in character_name