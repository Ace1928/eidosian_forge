from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def test_deprecated_decorator_with_explicit_thing(recwarn_always: pytest.WarningsRecorder) -> None:
    assert deprecated_with_thing() == 72
    got = recwarn_always.pop(TrioDeprecationWarning)
    assert isinstance(got.message, Warning)
    assert 'the thing is deprecated' in got.message.args[0]