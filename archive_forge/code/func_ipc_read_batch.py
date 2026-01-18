import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
def ipc_read_batch(buf):
    reader = pa.RecordBatchStreamReader(buf)
    return reader.read_next_batch()