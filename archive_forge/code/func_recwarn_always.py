from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
@pytest.fixture
def recwarn_always(recwarn: pytest.WarningsRecorder) -> pytest.WarningsRecorder:
    warnings.simplefilter('always')
    warnings.simplefilter('ignore', ResourceWarning)
    return recwarn