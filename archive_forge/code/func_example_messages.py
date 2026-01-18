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
@pytest.fixture
def example_messages(stream_fixture):
    batches = stream_fixture.write_batches()
    file_contents = stream_fixture.get_source()
    buf_reader = pa.BufferReader(file_contents)
    reader = pa.MessageReader.open_stream(buf_reader)
    return (batches, list(reader))