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
@pytest.mark.gzip
def test_message_read_from_compressed(example_messages):
    _, messages = example_messages
    for message in messages:
        raw_out = pa.BufferOutputStream()
        with pa.output_stream(raw_out, compression='gzip') as compressed_out:
            message.serialize_to(compressed_out)
        compressed_buf = raw_out.getvalue()
        result = pa.ipc.read_message(pa.input_stream(compressed_buf, compression='gzip'))
        assert result.equals(message)