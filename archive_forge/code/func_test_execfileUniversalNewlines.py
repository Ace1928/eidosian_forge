import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_execfileUniversalNewlines(self):
    """
        L{execfile} reads in the specified file using universal newlines so
        that scripts written on one platform will work on another.
        """
    for lineEnding in ('\n', '\r', '\r\n'):
        script = self.writeScript("foo = 'okay'" + lineEnding)
        globalNamespace = {'foo': None}
        execfile(script.path, globalNamespace)
        self.assertEqual('okay', globalNamespace['foo'])