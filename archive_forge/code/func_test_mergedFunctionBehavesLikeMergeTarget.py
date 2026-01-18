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
def test_mergedFunctionBehavesLikeMergeTarget(self):
    """
        After merging C{foo}'s data into C{bar}, the returned function behaves
        as if it is C{bar}.
        """
    foo_object = object()
    bar_object = object()

    def foo():
        return foo_object

    def bar(x, y, ab, c=10, *d, **e):
        a, b = ab
        return bar_object
    baz = util.mergeFunctionMetadata(foo, bar)
    self.assertIs(baz(1, 2, (3, 4), quux=10), bar_object)