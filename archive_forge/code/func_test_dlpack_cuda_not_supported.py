import ctypes
from functools import wraps
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
def test_dlpack_cuda_not_supported():
    cuda = pytest.importorskip('pyarrow.cuda')
    schema = pa.schema([pa.field('f0', pa.int16())])
    a0 = pa.array([1, 2, 3], type=pa.int16())
    batch = pa.record_batch([a0], schema=schema)
    cbuf = cuda.serialize_record_batch(batch, cuda.Context(0))
    cbatch = cuda.read_record_batch(cbuf, batch.schema)
    carr = cbatch['f0']
    with pytest.raises(NotImplementedError, match='DLPack support is implemented only for buffers on CPU device.'):
        np.from_dlpack(carr)
    with pytest.raises(NotImplementedError, match='DLPack support is implemented only for buffers on CPU device.'):
        carr.__dlpack_device__()