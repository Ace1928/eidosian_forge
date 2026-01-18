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
def test_array_constructor_from_pandas():
    import pandas as pd
    ext_type = IntegerType()
    storage = pa.array([1, 2, 3], type=pa.int64())
    expected = pa.ExtensionArray.from_storage(ext_type, storage)
    result = pa.array(pd.Series([1, 2, 3]), type=IntegerType())
    assert result.equals(expected)
    result = pa.array(pd.Series([1, 2, 3], dtype='category'), type=IntegerType())
    assert result.equals(expected)