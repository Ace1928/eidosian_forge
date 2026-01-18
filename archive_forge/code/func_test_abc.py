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
def test_abc(self):
    """
        L{util.InsensitiveDict} implements L{typing.MutableMapping}.
        """
    dct = util.InsensitiveDict()
    self.assertTrue(isinstance(dct, Iterable))
    self.assertTrue(isinstance(dct, Mapping))
    self.assertTrue(isinstance(dct, MutableMapping))
    self.assertFalse(isinstance(dct, Sequence))