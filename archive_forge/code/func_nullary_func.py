import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def nullary_func(context):
    return pa.array([42] * context.batch_length, type=pa.int64(), memory_pool=context.memory_pool)