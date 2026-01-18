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
def test_extra_info():
    with ErrorFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        try:
            list(client.do_action(flight.Action('protobuf', b'')))
            assert False
        except flight.FlightUnauthorizedError as e:
            assert e.extra_info is not None
            ei = e.extra_info
            assert ei == b'this is an error message'