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
@pytest.mark.slow
def test_cancel_do_get():
    """Test canceling a DoGet operation on the client side."""
    with ConstantFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        reader = client.do_get(flight.Ticket(b'ints'))
        reader.cancel()
        with pytest.raises(flight.FlightCancelledError, match='(?i).*cancel.*'):
            reader.read_chunk()