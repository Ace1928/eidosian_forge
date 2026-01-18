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
def testTriple(self):
    d = iter(util.IntervalDifferential([2, 4, 5], 10))
    for i in range(100):
        self.assertEqual(next(d), (2, 0))
        self.assertEqual(next(d), (2, 0))
        self.assertEqual(next(d), (0, 1))
        self.assertEqual(next(d), (1, 2))
        self.assertEqual(next(d), (1, 0))
        self.assertEqual(next(d), (2, 0))
        self.assertEqual(next(d), (0, 1))
        self.assertEqual(next(d), (2, 0))
        self.assertEqual(next(d), (0, 2))
        self.assertEqual(next(d), (2, 0))
        self.assertEqual(next(d), (0, 1))
        self.assertEqual(next(d), (2, 0))
        self.assertEqual(next(d), (1, 2))
        self.assertEqual(next(d), (1, 0))
        self.assertEqual(next(d), (0, 1))
        self.assertEqual(next(d), (2, 0))
        self.assertEqual(next(d), (2, 0))
        self.assertEqual(next(d), (0, 1))
        self.assertEqual(next(d), (0, 2))