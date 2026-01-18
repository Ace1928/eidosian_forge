import os
import random
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _test_dataframe
from pyarrow.tests.parquet.test_dataset import (
from pyarrow.util import guid
def test_file_context_manager(self):
    path = pjoin(self.tmp_path, 'ctx-manager')
    data = b'foo'
    with self.hdfs.open(path, 'wb') as f:
        f.write(data)
    with self.hdfs.open(path, 'rb') as f:
        assert f.size() == 3
        result = f.read(10)
        assert result == data