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
def test_middleware_reject():
    """Test rejecting an RPC with server middleware."""
    with HeaderFlightServer(middleware={'test': SelectiveAuthServerMiddlewareFactory()}) as server, FlightClient(('localhost', server.port)) as client:
        with pytest.raises(pa.ArrowNotImplementedError):
            list(client.list_actions())
        with pytest.raises(flight.FlightUnauthenticatedError):
            list(client.do_action(flight.Action(b'', b'')))
        client = FlightClient(('localhost', server.port), middleware=[SelectiveAuthClientMiddlewareFactory()])
        response = next(client.do_action(flight.Action(b'', b'')))
        assert b'password' == response.body.to_pybytes()