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
@skipIf(grp is None, 'Group Name/GID conversion requires the grp module.')
def test_gidFromGroupnameString(self):
    """
        When L{gidFromString} is called with a base-ten string representation
        of an integer, it returns the integer.
        """
    grent = grp.getgrgid(os.getgid())
    self.assertEqual(util.gidFromString(grent.gr_name), grent.gr_gid)