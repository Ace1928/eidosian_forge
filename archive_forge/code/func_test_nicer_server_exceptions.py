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
def test_nicer_server_exceptions():
    with ConvenienceServer() as server, FlightClient(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightServerError, match='a bytes-like object is required'):
            list(client.do_action('bad-action'))
        with pytest.raises(flight.FlightServerError, match='ArrowMemoryError'):
            list(client.do_action('arrow-exception'))