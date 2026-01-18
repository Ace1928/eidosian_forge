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
@contextlib.contextmanager
def enabled_auto_load():
    pa.PyExtensionType.set_auto_load(True)
    try:
        yield
    finally:
        pa.PyExtensionType.set_auto_load(False)