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
def test_never_sends_data():
    match = 'application server implementation error'
    with NeverSendsDataFlightServer() as server, flight.connect(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightServerError, match=match):
            client.do_get(flight.Ticket(b'')).read_all()
        table = client.do_get(flight.Ticket(b'yield_data')).read_all()
        assert table.num_rows == 5