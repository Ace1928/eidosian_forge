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
def is_arabic_isolated_form(character: str) -> bool:
    try:
        character_name = unicodedata.name(character)
    except ValueError:
        return False
    return 'ARABIC' in character_name and 'ISOLATED FORM' in character_name