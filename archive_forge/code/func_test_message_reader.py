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
def test_message_reader(example_messages):
    _, messages = example_messages
    assert len(messages) == 6
    assert messages[0].type == 'schema'
    assert isinstance(messages[0].metadata, pa.Buffer)
    assert isinstance(messages[0].body, pa.Buffer)
    assert messages[0].metadata_version == pa.MetadataVersion.V5
    for msg in messages[1:]:
        assert msg.type == 'record batch'
        assert isinstance(msg.metadata, pa.Buffer)
        assert isinstance(msg.body, pa.Buffer)
        assert msg.metadata_version == pa.MetadataVersion.V5