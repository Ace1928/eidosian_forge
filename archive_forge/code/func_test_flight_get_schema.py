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
def test_flight_get_schema():
    """Make sure GetSchema returns correct schema."""
    with GetInfoFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        info = client.get_schema(flight.FlightDescriptor.for_command(b''))
        assert info.schema == pa.schema([('a', pa.int32())])