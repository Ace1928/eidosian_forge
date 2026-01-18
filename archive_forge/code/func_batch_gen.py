import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def batch_gen(ctx):
    for n in range(3, 0, -1):
        yield _record_batch_for_range(schema, n - 1)
    raise RuntimeError('datasource1_exception')