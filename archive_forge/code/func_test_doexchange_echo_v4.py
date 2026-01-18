import ast
import base64
import itertools
import os
import pathlib
import signal
import struct
import tempfile
import threading
import time
import traceback
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.lib import IpcReadOptions, tobytes
from pyarrow.util import find_free_port
from pyarrow.tests import util
def test_doexchange_echo_v4():
    """Try a DoExchange echo server using the V4 metadata version."""
    data = pa.Table.from_arrays([pa.array(range(0, 10 * 1024))], names=['a'])
    batches = data.to_batches(max_chunksize=512)
    options = pa.ipc.IpcWriteOptions(metadata_version=pa.ipc.MetadataVersion.V4)
    with ExchangeFlightServer(options=options) as server, FlightClient(('localhost', server.port)) as client:
        descriptor = flight.FlightDescriptor.for_command(b'echo')
        writer, reader = client.do_exchange(descriptor)
        with writer:
            writer.begin(data.schema, options=options)
            for batch in batches:
                writer.write_batch(batch)
                assert reader.schema == data.schema
                chunk = reader.read_chunk()
                assert chunk.data == batch
                assert chunk.app_metadata is None