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
def test_flight_server_location_argument():
    locations = [None, 'grpc://localhost:0', ('localhost', find_free_port())]
    for location in locations:
        with FlightServerBase(location) as server:
            assert isinstance(server, FlightServerBase)