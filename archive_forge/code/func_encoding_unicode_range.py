import importlib
from codecs import IncrementalDecoder
from collections import Counter, OrderedDict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from .assets import FREQUENCIES
from .constant import KO_NAMES, LANGUAGE_SUPPORTED_COUNT, TOO_SMALL_SEQUENCE, ZH_NAMES
from .md import is_suspiciously_successive_range
from .models import CoherenceMatches
from .utils import (
def encoding_unicode_range(iana_name: str) -> List[str]:
    """
    Return associated unicode ranges in a single byte code page.
    """
    if is_multi_byte_encoding(iana_name):
        raise IOError('Function not supported on multi-byte code page')
    decoder = importlib.import_module('encodings.{}'.format(iana_name)).IncrementalDecoder
    p = decoder(errors='ignore')
    seen_ranges = {}
    character_count = 0
    for i in range(64, 255):
        chunk = p.decode(bytes([i]))
        if chunk:
            character_range = unicode_range(chunk)
            if character_range is None:
                continue
            if is_unicode_range_secondary(character_range) is False:
                if character_range not in seen_ranges:
                    seen_ranges[character_range] = 0
                seen_ranges[character_range] += 1
                character_count += 1
    return sorted([character_range for character_range in seen_ranges if seen_ranges[character_range] / character_count >= 0.15])