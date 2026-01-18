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
def test_immediate_cancel(self):
    """
        Immediately cancelling the result of @run_in_reactor function will
        still cancel the Deferred.
        """
    program = 'import os, threading, signal, time, sys\n\nfrom twisted.internet.defer import Deferred, CancelledError\n\nimport crochet\ncrochet.setup()\n\n@crochet.run_in_reactor\ndef run():\n    return Deferred()\n\ner = run()\ner.cancel()\ntry:\n    er.wait(1)\nexcept CancelledError:\n    sys.exit(23)\nelse:\n    sys.exit(3)\n'
    process = subprocess.Popen([sys.executable, '-c', program], cwd=crochet_directory)
    self.assertEqual(process.wait(), 23)