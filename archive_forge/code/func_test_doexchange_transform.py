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
def test_doexchange_transform():
    """Transform a table with a service."""
    data = pa.Table.from_arrays([pa.array(range(0, 1024)), pa.array(range(1, 1025)), pa.array(range(2, 1026))], names=['a', 'b', 'c'])
    expected = pa.Table.from_arrays([pa.array(range(3, 1024 * 3 + 3, 3))], names=['sum'])
    with ExchangeFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        descriptor = flight.FlightDescriptor.for_command(b'transform')
        writer, reader = client.do_exchange(descriptor)
        with writer:
            writer.begin(data.schema)
            writer.write_table(data)
            writer.done_writing()
            table = reader.read_all()
        assert expected == table