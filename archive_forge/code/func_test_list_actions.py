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
def test_list_actions():
    """Make sure the return type of ListActions is validated."""
    with ListActionsErrorFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightServerError, match='Results of list_actions must be ActionType or tuple'):
            list(client.list_actions())
    with ListActionsFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        assert list(client.list_actions()) == ListActionsFlightServer.expected_actions()