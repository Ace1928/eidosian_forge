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
def multiple_column_table():
    return pa.Table.from_arrays([pa.array(['foo', 'bar', 'baz', 'qux']), pa.array([1, 2, 3, 4])], names=['a', 'b'])