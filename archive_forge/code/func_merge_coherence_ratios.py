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
def merge_coherence_ratios(results: List[CoherenceMatches]) -> CoherenceMatches:
    """
    This function merge results previously given by the function coherence_ratio.
    The return type is the same as coherence_ratio.
    """
    per_language_ratios = OrderedDict()
    for result in results:
        for sub_result in result:
            language, ratio = sub_result
            if language not in per_language_ratios:
                per_language_ratios[language] = [ratio]
                continue
            per_language_ratios[language].append(ratio)
    merge = [(language, round(sum(per_language_ratios[language]) / len(per_language_ratios[language]), 4)) for language in per_language_ratios]
    return sorted(merge, key=lambda x: x[1], reverse=True)