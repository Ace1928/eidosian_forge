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
def test_doexchange_put():
    """Emulate DoPut with DoExchange."""
    data = pa.Table.from_arrays([pa.array(range(0, 10 * 1024))], names=['a'])
    batches = data.to_batches(max_chunksize=512)
    with ExchangeFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        descriptor = flight.FlightDescriptor.for_command(b'put')
        writer, reader = client.do_exchange(descriptor)
        with writer:
            writer.begin(data.schema)
            for batch in batches:
                writer.write_batch(batch)
            writer.done_writing()
            chunk = reader.read_chunk()
            assert chunk.data is None
            expected_buf = str(len(batches)).encode('utf-8')
            assert chunk.app_metadata == expected_buf