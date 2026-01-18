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
def test_interpreter_shutdown():
    """
    Ensure that the gRPC server is stopped at interpreter shutdown.

    See https://issues.apache.org/jira/browse/ARROW-16597.
    """
    util.invoke_script('arrow_16597.py')