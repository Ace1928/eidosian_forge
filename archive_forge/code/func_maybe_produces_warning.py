from __future__ import annotations
from contextlib import (
import inspect
import re
import sys
from typing import (
import warnings
from pandas.compat import PY311
def maybe_produces_warning(warning: type[Warning], condition: bool, **kwargs):
    """
    Return a context manager that possibly checks a warning based on the condition
    """
    if condition:
        return assert_produces_warning(warning, **kwargs)
    else:
        return nullcontext()