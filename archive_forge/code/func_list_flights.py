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
def list_flights(self, context, criteria):
    yield flight.FlightInfo(pa.schema([]), flight.FlightDescriptor.for_path('/foo'), [], -1, -1)
    raise flight.FlightInternalError('foo')