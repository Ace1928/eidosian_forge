from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_construct_ragged_dtype():
    dtype = RaggedDtype()
    assert dtype.type == np.ndarray
    assert dtype.name == 'Ragged[{subtype}]'.format(subtype=dtype.subtype)
    assert dtype.kind == 'O'