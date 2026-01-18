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
def test_currentUID(self):
    """
        If the current uid is the same as the uid passed to L{util.switchUID},
        then initgroups does not get called, but a warning is issued.
        """
    uid = self.mockos.getuid()
    util.switchUID(uid, None)
    self.assertEqual(self.initgroupsCalls, [])
    self.assertEqual(self.mockos.actions, [])
    currentWarnings = self.flushWarnings([util.switchUID])
    self.assertEqual(len(currentWarnings), 1)
    self.assertIn('tried to drop privileges and setuid %i' % uid, currentWarnings[0]['message'])
    self.assertIn('but uid is already %i' % uid, currentWarnings[0]['message'])