import fractions
import platform
import types
from typing import Any, Type
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, IS_MUSL
def test_abc_complexfloating(self) -> None:
    alias = np.complexfloating[Any, Any]
    assert isinstance(alias, types.GenericAlias)
    assert alias.__origin__ is np.complexfloating