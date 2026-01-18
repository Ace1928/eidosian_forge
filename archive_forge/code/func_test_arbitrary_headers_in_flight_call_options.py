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
def test_arbitrary_headers_in_flight_call_options():
    """Test passing multiple arbitrary headers to the middleware."""
    with ArbitraryHeadersFlightServer(auth_handler=no_op_auth_handler, middleware={'auth': HeaderAuthServerMiddlewareFactory(), 'arbitrary-headers': ArbitraryHeadersServerMiddlewareFactory()}) as server, FlightClient(('localhost', server.port)) as client:
        token_pair = client.authenticate_basic_token(b'test', b'password')
        assert token_pair[0] == b'authorization'
        assert token_pair[1] == b'Bearer token1234'
        options = flight.FlightCallOptions(headers=[token_pair, (b'test-header-1', b'value1'), (b'test-header-2', b'value2')])
        result = list(client.do_action(flight.Action('test-action', b''), options=options))
        assert result[0].body.to_pybytes() == b'value1'
        assert result[1].body.to_pybytes() == b'value2'