import warnings
from collections import Counter
from encodings.aliases import aliases
from hashlib import sha256
from json import dumps
from re import sub
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .constant import NOT_PRINTABLE_PATTERN, TOO_BIG_SEQUENCE
from .md import mess_ratio
from .utils import iana_name, is_multi_byte_encoding, unicode_range
@property
def coherence_non_latin(self) -> float:
    """
        Coherence ratio on the first non-latin language detected if ANY.
        Notice: Will be removed in 3.0
        """
    warnings.warn('coherence_non_latin is deprecated and will be removed in 3.0', DeprecationWarning)
    return 0.0