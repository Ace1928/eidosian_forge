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
def alphabet_languages(characters: List[str], ignore_non_latin: bool=False) -> List[str]:
    """
    Return associated languages associated to given characters.
    """
    languages = []
    source_have_accents = any((is_accentuated(character) for character in characters))
    for language, language_characters in FREQUENCIES.items():
        target_have_accents, target_pure_latin = get_target_features(language)
        if ignore_non_latin and target_pure_latin is False:
            continue
        if target_have_accents is False and source_have_accents:
            continue
        character_count = len(language_characters)
        character_match_count = len([c for c in language_characters if c in characters])
        ratio = character_match_count / character_count
        if ratio >= 0.2:
            languages.append((language, ratio))
    languages = sorted(languages, key=lambda x: x[1], reverse=True)
    return [compatible_language[0] for compatible_language in languages]