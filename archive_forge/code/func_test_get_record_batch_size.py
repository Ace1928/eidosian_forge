from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
@pytest.mark.pandas
def test_get_record_batch_size():
    N = 10
    itemsize = 8
    df = pd.DataFrame({'foo': np.random.randn(N)})
    batch = pa.RecordBatch.from_pandas(df)
    assert pa.ipc.get_record_batch_size(batch) > N * itemsize