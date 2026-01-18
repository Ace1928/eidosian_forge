import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def other_process_for_test_IPC(handle_buffer, expected_arr):
    other_context = pa.cuda.Context(0)
    ipc_handle = pa.cuda.IpcMemHandle.from_buffer(handle_buffer)
    ipc_buf = other_context.open_ipc_buffer(ipc_handle)
    ipc_buf.context.synchronize()
    buf = ipc_buf.copy_to_host()
    assert buf.size == expected_arr.size, repr((buf.size, expected_arr.size))
    arr = np.frombuffer(buf, dtype=expected_arr.dtype)
    np.testing.assert_equal(arr, expected_arr)