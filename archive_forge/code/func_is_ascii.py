import importlib
import logging
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder  # type: ignore
from .constant import (
def is_ascii(character: str) -> bool:
    try:
        character.encode('ascii')
    except UnicodeEncodeError:
        return False
    return True