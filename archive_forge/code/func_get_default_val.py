from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def get_default_val(pat: str):
    key = _get_single_key(pat, silent=True)
    return _get_registered_option(key).defval