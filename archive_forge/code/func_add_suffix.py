from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def add_suffix(x: Any, suffix: Any) -> Any:
    x['completion'] += suffix
    return x