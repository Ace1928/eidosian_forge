import os
import pytest
import sys
from tempfile import TemporaryFile
from numpy.distutils import exec_command
from numpy.distutils.exec_command import get_pythonexe
from numpy.testing import tempdir, assert_, assert_warns, IS_WASM
from io import StringIO
def setup_method(self):
    self.pyexe = get_pythonexe()