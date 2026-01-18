from __future__ import annotations
from functools import wraps
import inspect
from textwrap import dedent
from typing import (
import warnings
from pandas._libs.properties import cache_readonly
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def future_version_msg(version: str | None) -> str:
    """Specify which version of pandas the deprecation will take place in."""
    if version is None:
        return 'In a future version of pandas'
    else:
        return f'Starting with pandas version {version}'