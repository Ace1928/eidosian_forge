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
def test_file_pathlib(file_fixture, tmpdir):
    file_fixture.write_batches()
    source = file_fixture.get_source()
    path = tmpdir.join('file.arrow').strpath
    with open(path, 'wb') as f:
        f.write(source)
    t1 = pa.ipc.open_file(pathlib.Path(path)).read_all()
    t2 = pa.ipc.open_file(pa.OSFile(path)).read_all()
    assert t1.equals(t2)