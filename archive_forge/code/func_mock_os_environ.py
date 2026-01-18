import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
@contextlib.contextmanager
def mock_os_environ(args):
    old_environ = os.environ
    os.environ = args
    try:
        yield
    finally:
        os.environ = old_environ