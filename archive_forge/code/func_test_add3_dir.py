import gc
import glob
import os
import shutil
import sys
import tempfile
from io import StringIO
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.common.tempfiles as tempfiles
from pyomo.common.dependencies import pyutilib_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import (
def test_add3_dir(self):
    """Test explicit adding of a directory that already exists"""
    os.mkdir(tempdir + 'add3')
    TempfileManager.add_tempfile(tempdir + 'add3')