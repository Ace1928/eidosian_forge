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
def resource_root():
    """Get the path to the test resources directory."""
    if not os.environ.get('ARROW_TEST_DATA'):
        raise RuntimeError('Test resources not found; set ARROW_TEST_DATA to <repo root>/testing/data')
    return pathlib.Path(os.environ['ARROW_TEST_DATA']) / 'flight'