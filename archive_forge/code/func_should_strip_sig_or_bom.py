import importlib
import logging
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder  # type: ignore
from .constant import (
def should_strip_sig_or_bom(iana_encoding: str) -> bool:
    return iana_encoding not in {'utf_16', 'utf_32'}