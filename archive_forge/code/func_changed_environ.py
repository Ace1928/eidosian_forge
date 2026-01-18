import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
@contextlib.contextmanager
def changed_environ(name, value):
    """
    Temporarily set environment variable *name* to *value*.
    """
    orig_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if orig_value is None:
            del os.environ[name]
        else:
            os.environ[name] = orig_value