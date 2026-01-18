import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_codecsOpenText(self):
    """
        When passed an encoding, however, the L{codecs} module returns unicode.
        """
    with codecs.open(self.mktemp(), 'wb', encoding='utf-8') as f:
        self.assertEqual(ioType(f), str)