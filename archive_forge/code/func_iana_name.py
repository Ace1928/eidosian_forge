import importlib
import logging
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder  # type: ignore
from .constant import (
def iana_name(cp_name: str, strict: bool=True) -> str:
    cp_name = cp_name.lower().replace('-', '_')
    for encoding_alias, encoding_iana in aliases.items():
        if cp_name in [encoding_alias, encoding_iana]:
            return encoding_iana
    if strict:
        raise ValueError("Unable to retrieve IANA for '{}'".format(cp_name))
    return cp_name