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
@lru_cache()
def mb_encoding_languages(iana_name: str) -> List[str]:
    """
    Multi-byte encoding language association. Some code page are heavily linked to particular language(s).
    This function does the correspondence.
    """
    if iana_name.startswith('shift_') or iana_name.startswith('iso2022_jp') or iana_name.startswith('euc_j') or (iana_name == 'cp932'):
        return ['Japanese']
    if iana_name.startswith('gb') or iana_name in ZH_NAMES:
        return ['Chinese', 'Classical Chinese']
    if iana_name.startswith('iso2022_kr') or iana_name in KO_NAMES:
        return ['Korean']
    return []