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
def characters_popularity_compare(language: str, ordered_characters: List[str]) -> float:
    """
    Determine if a ordered characters list (by occurrence from most appearance to rarest) match a particular language.
    The result is a ratio between 0. (absolutely no correspondence) and 1. (near perfect fit).
    Beware that is function is not strict on the match in order to ease the detection. (Meaning close match is 1.)
    """
    if language not in FREQUENCIES:
        raise ValueError('{} not available'.format(language))
    character_approved_count = 0
    for character in ordered_characters:
        if character not in FREQUENCIES[language]:
            continue
        characters_before_source = FREQUENCIES[language][0:FREQUENCIES[language].index(character)]
        characters_after_source = FREQUENCIES[language][FREQUENCIES[language].index(character):]
        characters_before = ordered_characters[0:ordered_characters.index(character)]
        characters_after = ordered_characters[ordered_characters.index(character):]
        before_match_count = [e in characters_before for e in characters_before_source].count(True)
        after_match_count = [e in characters_after for e in characters_after_source].count(True)
        if len(characters_before_source) == 0 and before_match_count <= 4:
            character_approved_count += 1
            continue
        if len(characters_after_source) == 0 and after_match_count <= 4:
            character_approved_count += 1
            continue
        if before_match_count / len(characters_before_source) >= 0.4 or after_match_count / len(characters_after_source) >= 0.4:
            character_approved_count += 1
            continue
    return character_approved_count / len(ordered_characters)