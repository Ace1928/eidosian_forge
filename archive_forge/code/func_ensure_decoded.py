from __future__ import annotations
from functools import reduce
import numpy as np
from pandas._config import get_option
def ensure_decoded(s) -> str:
    """
    If we have bytes, decode them to unicode.
    """
    if isinstance(s, (np.bytes_, bytes)):
        s = s.decode(get_option('display.encoding'))
    return s