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
def testRemove(self):
    d = iter(util.IntervalDifferential([3, 5], 10))
    self.assertEqual(next(d), (3, 0))
    self.assertEqual(next(d), (2, 1))
    self.assertEqual(next(d), (1, 0))
    d.removeInterval(3)
    self.assertEqual(next(d), (4, 0))
    self.assertEqual(next(d), (5, 0))
    d.removeInterval(5)
    self.assertEqual(next(d), (10, None))
    self.assertRaises(ValueError, d.removeInterval, 10)