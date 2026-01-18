import errno
import os.path
import shutil
import sys
import warnings
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.python import util
from twisted.python.filepath import FilePath
from twisted.test.test_process import MockOS
from twisted.trial.unittest import FailTest, TestCase
from twisted.trial.util import suppress as SUPPRESS
def test_getPluginDirs(self):
    """
        L{util.getPluginDirs} is deprecated.
        """
    util.getPluginDirs()
    currentWarnings = self.flushWarnings(offendingFunctions=[self.test_getPluginDirs])
    self.assertEqual(currentWarnings[0]['message'], 'twisted.python.util.getPluginDirs is deprecated since Twisted 12.2.')
    self.assertEqual(currentWarnings[0]['category'], DeprecationWarning)
    self.assertEqual(len(currentWarnings), 1)