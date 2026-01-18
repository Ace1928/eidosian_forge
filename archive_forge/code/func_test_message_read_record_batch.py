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
def test_message_read_record_batch(example_messages):
    batches, messages = example_messages
    for batch, message in zip(batches, messages[1:]):
        read_batch = pa.ipc.read_record_batch(message, batch.schema)
        assert read_batch.equals(batch)