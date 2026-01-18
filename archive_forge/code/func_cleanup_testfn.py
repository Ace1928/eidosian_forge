import io
import distutils.core
import os
import shutil
import sys
from test.support import captured_stdout
from test.support import os_helper
import unittest
from distutils.tests import support
from distutils import log
from distutils.core import setup
import os
from distutils.core import setup
from distutils.core import setup
from distutils.core import setup
from distutils.command.install import install as _install
def cleanup_testfn(self):
    path = os_helper.TESTFN
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)