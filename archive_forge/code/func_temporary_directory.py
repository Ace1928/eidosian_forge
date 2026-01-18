import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
@contextlib.contextmanager
def temporary_directory():
    """
    Context manager to create and clean up a temporary directory.

    """
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)