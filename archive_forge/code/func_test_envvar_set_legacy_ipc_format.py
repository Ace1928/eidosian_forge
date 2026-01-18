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
def test_envvar_set_legacy_ipc_format():
    schema = pa.schema([pa.field('foo', pa.int32())])
    writer = pa.ipc.new_stream(pa.BufferOutputStream(), schema)
    assert not writer._use_legacy_format
    assert writer._metadata_version == pa.ipc.MetadataVersion.V5
    writer = pa.ipc.new_file(pa.BufferOutputStream(), schema)
    assert not writer._use_legacy_format
    assert writer._metadata_version == pa.ipc.MetadataVersion.V5
    with changed_environ('ARROW_PRE_0_15_IPC_FORMAT', '1'):
        writer = pa.ipc.new_stream(pa.BufferOutputStream(), schema)
        assert writer._use_legacy_format
        assert writer._metadata_version == pa.ipc.MetadataVersion.V5
        writer = pa.ipc.new_file(pa.BufferOutputStream(), schema)
        assert writer._use_legacy_format
        assert writer._metadata_version == pa.ipc.MetadataVersion.V5
    with changed_environ('ARROW_PRE_1_0_METADATA_VERSION', '1'):
        writer = pa.ipc.new_stream(pa.BufferOutputStream(), schema)
        assert not writer._use_legacy_format
        assert writer._metadata_version == pa.ipc.MetadataVersion.V4
        writer = pa.ipc.new_file(pa.BufferOutputStream(), schema)
        assert not writer._use_legacy_format
        assert writer._metadata_version == pa.ipc.MetadataVersion.V4
    with changed_environ('ARROW_PRE_1_0_METADATA_VERSION', '1'):
        with changed_environ('ARROW_PRE_0_15_IPC_FORMAT', '1'):
            writer = pa.ipc.new_stream(pa.BufferOutputStream(), schema)
            assert writer._use_legacy_format
            assert writer._metadata_version == pa.ipc.MetadataVersion.V4
            writer = pa.ipc.new_file(pa.BufferOutputStream(), schema)
            assert writer._use_legacy_format
            assert writer._metadata_version == pa.ipc.MetadataVersion.V4