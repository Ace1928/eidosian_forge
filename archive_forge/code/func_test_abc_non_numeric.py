import fractions
import platform
import types
from typing import Any, Type
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, IS_MUSL
@pytest.mark.parametrize('cls', [np.generic, np.flexible, np.character])
def test_abc_non_numeric(self, cls: Type[np.generic]) -> None:
    with pytest.raises(TypeError):
        cls[Any]