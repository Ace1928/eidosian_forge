import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
@pytest.mark.pandas
def test_ext_array_conversion_to_pandas():
    import pandas as pd
    storage1 = pa.array([1, 2, 3], type=pa.int64())
    storage2 = pa.array([b'123', b'456', b'789'], type=pa.binary(3))
    ty1 = IntegerType()
    ty2 = ParamExtType(3)
    arr1 = pa.ExtensionArray.from_storage(ty1, storage1)
    arr2 = pa.ExtensionArray.from_storage(ty2, storage2)
    result = arr1.to_pandas()
    expected = pd.Series([1, 2, 3], dtype='int64')
    pd.testing.assert_series_equal(result, expected)
    result = arr2.to_pandas()
    expected = pd.Series([b'123', b'456', b'789'], dtype=object)
    pd.testing.assert_series_equal(result, expected)