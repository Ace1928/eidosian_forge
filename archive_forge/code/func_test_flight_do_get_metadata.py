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
def test_flight_do_get_metadata():
    """Try a simple do_get call with metadata."""
    data = [pa.array([-10, -5, 0, 5, 10])]
    table = pa.Table.from_arrays(data, names=['a'])
    batches = []
    with MetadataFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        reader = client.do_get(flight.Ticket(b''))
        idx = 0
        while True:
            try:
                batch, metadata = reader.read_chunk()
                batches.append(batch)
                server_idx, = struct.unpack('<i', metadata.to_pybytes())
                assert idx == server_idx
                idx += 1
            except StopIteration:
                break
        data = pa.Table.from_batches(batches)
        assert data.equals(table)