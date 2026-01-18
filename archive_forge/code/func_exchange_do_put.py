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
def exchange_do_put(self, context, reader, writer):
    """Emulate DoPut with DoExchange."""
    num_batches = 0
    for chunk in reader:
        if not chunk.data:
            raise pa.ArrowInvalid('All chunks must have data.')
        num_batches += 1
    writer.write_metadata(str(num_batches).encode('utf-8'))