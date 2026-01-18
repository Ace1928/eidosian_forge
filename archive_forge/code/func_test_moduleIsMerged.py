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
def test_moduleIsMerged(self):
    """
        Merging C{foo} into C{bar} returns a function with C{foo}'s
        C{__module__}.
        """

    def foo():
        pass

    def bar():
        pass
    bar.__module__ = 'somewhere.else'
    baz = util.mergeFunctionMetadata(foo, bar)
    self.assertEqual(baz.__module__, foo.__module__)