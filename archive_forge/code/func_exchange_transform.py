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
def exchange_transform(self, context, reader, writer):
    """Sum rows in an uploaded table."""
    for field in reader.schema:
        if not pa.types.is_integer(field.type):
            raise pa.ArrowInvalid('Invalid field: ' + repr(field))
    table = reader.read_all()
    sums = [0] * table.num_rows
    for column in table:
        for row, value in enumerate(column):
            sums[row] += value.as_py()
    result = pa.Table.from_arrays([pa.array(sums)], names=['sum'])
    writer.begin(result.schema)
    writer.write_table(result)