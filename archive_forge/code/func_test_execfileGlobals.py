import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_execfileGlobals(self):
    """
        L{execfile} executes the specified file in the given global namespace.
        """
    script = self.writeScript('foo += 1\n')
    globalNamespace = {'foo': 1}
    execfile(script.path, globalNamespace)
    self.assertEqual(2, globalNamespace['foo'])