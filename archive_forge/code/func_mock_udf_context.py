import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
def mock_udf_context(batch_length=10):
    from pyarrow._compute import _get_udf_context
    return _get_udf_context(pa.default_memory_pool(), batch_length)