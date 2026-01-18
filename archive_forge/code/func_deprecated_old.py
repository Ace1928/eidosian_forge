from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
@deprecated('1.5', issue=123, instead=new)
def deprecated_old() -> int:
    return 3