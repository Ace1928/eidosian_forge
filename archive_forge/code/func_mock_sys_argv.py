import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
@contextlib.contextmanager
def mock_sys_argv(args):
    old_args = sys.argv
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = old_args