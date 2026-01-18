from __future__ import annotations
import datetime as dt
import os
import sys
from typing import Any, Iterable
import numpy as np
import param
def isurl(obj: Any, formats: Iterable[str] | None=None) -> bool:
    if not isinstance(obj, str):
        return False
    lower_string = obj.lower().split('?')[0].split('#')[0]
    return lower_string.startswith(('http://', 'https://')) and (formats is None or any((lower_string.endswith('.' + fmt) for fmt in formats)))