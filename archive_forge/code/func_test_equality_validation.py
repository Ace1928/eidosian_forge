from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
@pytest.mark.parametrize('other', ['a string', 32, RaggedArray([[0, 1], [2, 3, 4]]), np.array([[0, 1], [2, 3, 4]], dtype='object'), np.array([[0, 1], [2, 3]], dtype='int32')])
def test_equality_validation(other):
    arg1 = [[1, 2], [], [1, 2], None, [11, 22, 33, 44]]
    ra1 = RaggedArray(arg1, dtype='int32')
    with pytest.raises(ValueError, match='Cannot check equality'):
        ra1 == other