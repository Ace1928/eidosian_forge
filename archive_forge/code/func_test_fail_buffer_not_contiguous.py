import h5py
import numpy
import numpy.testing
import pytest
from .common import ut, TestCase
def test_fail_buffer_not_contiguous(self, writable_file):
    ref_data = numpy.arange(16).reshape(4, 4)
    dataset = writable_file.create_dataset('uncompressed', data=ref_data, chunks=ref_data.shape)
    array = numpy.empty(ref_data.shape + (2,), dtype=ref_data.dtype)
    out = array[:, :, ::2]
    with pytest.raises(ValueError):
        dataset.id.read_direct_chunk((0, 0), out=out)