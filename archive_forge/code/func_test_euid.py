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
def test_euid(self):
    """
        L{util.switchUID} calls L{util.initgroups} and then C{os.seteuid} with
        the given uid if the C{euid} parameter is set to C{True}.
        """
    util.switchUID(12000, None, True)
    self.assertEqual(self.initgroupsCalls, [(12000, None)])
    self.assertEqual(self.mockos.seteuidCalls, [12000])