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
def test_do_action_result_convenience():
    with ConvenienceServer() as server, FlightClient(('localhost', server.port)) as client:
        results = [x.body for x in client.do_action('simple-action')]
        assert results == server.simple_action_results
        body = b'the-body'
        results = [x.body for x in client.do_action(('echo', body))]
        assert results == [body]