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
def read_flight_resource(path):
    """Get the contents of a test resource file."""
    root = resource_root()
    if not root:
        return None
    try:
        with (root / path).open('rb') as f:
            return f.read()
    except FileNotFoundError:
        raise RuntimeError('Test resource {} not found; did you initialize the test resource submodule?\n{}'.format(root / path, traceback.format_exc()))