import json
import os
import re
import unicodedata
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def ensure_text(s, encoding='utf-8', errors='strict'):
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    elif isinstance(s, str):
        return s
    else:
        raise TypeError(f"not expecting type '{type(s)}'")