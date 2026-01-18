import os
import pytest
import sys
from tempfile import TemporaryFile
from numpy.distutils import exec_command
from numpy.distutils.exec_command import get_pythonexe
from numpy.testing import tempdir, assert_, assert_warns, IS_WASM
from io import StringIO
def test_basic(self):
    with redirect_stdout(StringIO()):
        with redirect_stderr(StringIO()):
            with assert_warns(DeprecationWarning):
                if os.name == 'posix':
                    self.check_posix(use_tee=0)
                    self.check_posix(use_tee=1)
                elif os.name == 'nt':
                    self.check_nt(use_tee=0)
                    self.check_nt(use_tee=1)
                self.check_execute_in(use_tee=0)
                self.check_execute_in(use_tee=1)