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
def test_inheritedClassesInequality(self):
    """
        An instance of a class which derives from a class which mixes in
        L{FancyEqMixin} should compare unequal to an instance of the base
        class if any of their attributes compare unequal.
        """
    self.assertFalse(Record(1, 2) != DerivedRecord(1, 2))
    self.assertTrue(Record(1, 2) != DerivedRecord(1, 3))
    self.assertTrue(Record(1, 2) != DerivedRecord(2, 2))
    self.assertTrue(Record(1, 2) != DerivedRecord(3, 4))