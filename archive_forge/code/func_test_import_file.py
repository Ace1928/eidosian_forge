import ctypes
import logging
import os
import platform
import shutil
import stat
import sys
import tempfile
import subprocess
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.common.envvar as envvar
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import (
from pyomo.common.download import FileDownloader
def test_import_file(self):
    import_ex = import_file(os.path.join(_this_file_dir, 'import_ex.py'))
    self.assertIn('pyomo.common.tests.import_ex', sys.modules)
    self.assertEqual(import_ex.b, 2)