import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def test_Context():
    assert cuda.Context.get_num_devices() > 0
    assert global_context.device_number == 0
    assert global_context1.device_number == cuda.Context.get_num_devices() - 1
    with pytest.raises(ValueError, match='device_number argument must be non-negative less than'):
        cuda.Context(cuda.Context.get_num_devices())