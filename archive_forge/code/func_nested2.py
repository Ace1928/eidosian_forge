from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def nested2() -> None:
    warn_deprecated('x', '1.3', issue=7, instead='y', stacklevel=3)