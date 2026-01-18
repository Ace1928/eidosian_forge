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
def test_noPreserve(self):
    """
        L{util.InsensitiveDict} does not preserves the case of keys if
        constructed with C{preserve=False}.
        """
    dct = util.InsensitiveDict({'Foo': 'bar', 1: 2, 'fnz': {1: 2}}, preserve=0)
    keys = ['foo', 'fnz', 1]
    for x in keys:
        self.assertIn(x, dct.keys())
        self.assertIn((x, dct[x]), dct.items())
    self.assertEqual(len(keys), len(dct))
    del dct[1]
    del dct['foo']
    self.assertEqual(dct.keys(), ['fnz'])