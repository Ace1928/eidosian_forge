import pyarrow as pa
from pyarrow import filesystem
import os
import pytest
def test_filesystem_deprecated_toplevel():
    with pytest.warns(FutureWarning):
        pa.localfs
    with pytest.warns(FutureWarning):
        pa.FileSystem
    with pytest.warns(FutureWarning):
        pa.LocalFileSystem
    with pytest.warns(FutureWarning):
        pa.HadoopFileSystem