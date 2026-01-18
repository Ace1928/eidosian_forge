from __future__ import absolute_import
import threading
import subprocess
import time
import gc
import sys
import weakref
import tempfile
import os
import inspect
from unittest import SkipTest
from twisted.trial.unittest import TestCase
from twisted.internet.defer import succeed, Deferred, fail, CancelledError
from twisted.python.failure import Failure
from twisted.python import threadable
from twisted.python.runtime import platform
from .._eventloop import (
from .test_setup import FakeReactor
from .. import (
from ..tests import crochet_directory
import os, threading, signal, time, sys
import crochet
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred, CancelledError
import crochet
from crochet import EventualResult
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
import crochet
from twisted.internet.defer import Deferred
import os, threading, signal, time, sys
from twisted.internet.defer import Deferred
from twisted.internet import reactor
import crochet
def test_noWaitingDuringImport(self):
    """
        EventualResult.wait() raises an exception if called while a module is
        being imported.

        This prevents the imports from taking a long time, preventing other
        imports from running in other threads. It also prevents deadlocks,
        which can happen if the code being waited on also tries to import
        something.
        """
    if sys.version_info[0] > 2:
        from unittest import SkipTest
        raise SkipTest('This test is too fragile (and insufficient) on Python 3 - see https://github.com/itamarst/crochet/issues/43')
    directory = tempfile.mktemp()
    os.mkdir(directory)
    sys.path.append(directory)
    self.addCleanup(sys.path.remove, directory)
    with open(os.path.join(directory, 'shouldbeunimportable.py'), 'w') as f:
        f.write('from crochet import EventualResult\nfrom twisted.internet.defer import Deferred\n\nEventualResult(Deferred(), None).wait(1.0)\n')
    self.assertRaises(RuntimeError, __import__, 'shouldbeunimportable')