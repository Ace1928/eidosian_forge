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
def test_middleware_multi_header():
    """Test sending/receiving multiple (binary-valued) headers."""
    with MultiHeaderFlightServer(middleware={'test': MultiHeaderServerMiddlewareFactory()}) as server:
        headers = MultiHeaderClientMiddlewareFactory()
        with FlightClient(('localhost', server.port), middleware=[headers]) as client:
            response = next(client.do_action(flight.Action(b'', b'')))
            raw_headers = response.body.to_pybytes().decode('utf-8')
            client_headers = ast.literal_eval(raw_headers)
            for header, values in MultiHeaderClientMiddleware.EXPECTED.items():
                header = header.lower()
                if isinstance(header, bytes):
                    header = header.decode('ascii')
                assert client_headers.get(header) == values
                assert headers.last_headers.get(header) == values