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
def test_UID(self):
    """
        Check UID/GID switches when current effective UID is non-root.
        """
    self._testUIDGIDSwitch(1, 0, 0, 0, [0, 1], [])
    self._testUIDGIDSwitch(1, 0, 1, 0, [], [])
    self._testUIDGIDSwitch(1, 0, 1, 1, [0, 1, 0, 1], [1, 0])
    self._testUIDGIDSwitch(1, 0, 2, 1, [0, 2, 0, 1], [1, 0])