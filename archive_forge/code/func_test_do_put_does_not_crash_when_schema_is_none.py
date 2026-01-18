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
def test_do_put_does_not_crash_when_schema_is_none():
    client = FlightClient('grpc+tls://localhost:9643', disable_server_verification=True)
    msg = "Argument 'schema' has incorrect type \\(expected pyarrow.lib.Schema, got NoneType\\)"
    with pytest.raises(TypeError, match=msg):
        client.do_put(flight.FlightDescriptor.for_command('foo'), schema=None)