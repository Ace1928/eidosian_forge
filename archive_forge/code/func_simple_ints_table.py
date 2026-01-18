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
def simple_ints_table():
    data = [pa.array([-10, -5, 0, 5, 10])]
    return pa.Table.from_arrays(data, names=['some_ints'])