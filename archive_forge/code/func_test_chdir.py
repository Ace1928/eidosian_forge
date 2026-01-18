import glob
import operator
import os
import shutil
import sys
import tempfile
from incremental import Version
from twisted.python import release
from twisted.python._release import (
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def test_chdir(self):
    """
        Test that the runChdirSafe is actually safe, i.e., it still
        changes back to the original directory even if an error is
        raised.
        """
    cwd = os.getcwd()

    def chAndBreak():
        os.mkdir('releaseCh')
        os.chdir('releaseCh')
        1 // 0
    self.assertRaises(ZeroDivisionError, release.runChdirSafe, chAndBreak)
    self.assertEqual(cwd, os.getcwd())