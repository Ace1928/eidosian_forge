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
def alpha_unicode_split(decoded_sequence: str) -> List[str]:
    """
    Given a decoded text sequence, return a list of str. Unicode range / alphabet separation.
    Ex. a text containing English/Latin with a bit a Hebrew will return two items in the resulting list;
    One containing the latin letters and the other hebrew.
    """
    layers = OrderedDict()
    for character in decoded_sequence:
        if character.isalpha() is False:
            continue
        character_range = unicode_range(character)
        if character_range is None:
            continue
        layer_target_range = None
        for discovered_range in layers:
            if is_suspiciously_successive_range(discovered_range, character_range) is False:
                layer_target_range = discovered_range
                break
        if layer_target_range is None:
            layer_target_range = character_range
        if layer_target_range not in layers:
            layers[layer_target_range] = character.lower()
            continue
        layers[layer_target_range] += character.lower()
    return list(layers.values())