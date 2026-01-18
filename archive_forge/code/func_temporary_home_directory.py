import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
@contextlib.contextmanager
def temporary_home_directory():
    """
    Context manager that temporarily remaps HOME / APPDATA
    to a temporary directory.

    """
    home_var = 'APPDATA' if sys.platform == 'win32' else 'HOME'
    with temporary_directory() as temp_home:
        with restore_mapping_entry(os.environ, home_var):
            os.environ[home_var] = temp_home
            yield