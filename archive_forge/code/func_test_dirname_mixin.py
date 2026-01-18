from collections.abc import Generator
import contextlib
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import accessor
def test_dirname_mixin() -> None:

    class X(accessor.DirNamesMixin):
        x = 1
        y: int

        def __init__(self) -> None:
            self.z = 3
    result = [attr_name for attr_name in dir(X()) if not attr_name.startswith('_')]
    assert result == ['x', 'z']