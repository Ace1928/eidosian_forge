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
def test_control_c_is_possible(self):
    """
        A call to a decorated function responds to a Ctrl-C (i.e. with a
        KeyboardInterrupt) in a timely manner.
        """
    if platform.type != 'posix':
        raise SkipTest("I don't have the energy to fight Windows semantics.")
    program = "import os, threading, signal, time, sys\nimport crochet\ncrochet.setup()\nfrom twisted.internet.defer import Deferred\n\nif sys.platform.startswith('win'):\n    signal.signal(signal.SIGBREAK, signal.default_int_handler)\n    sig_int=signal.CTRL_BREAK_EVENT\n    sig_kill=signal.SIGTERM\nelse:\n    sig_int=signal.SIGINT\n    sig_kill=signal.SIGKILL\n\n\ndef interrupt():\n    time.sleep(0.1) # Make sure we've hit wait()\n    os.kill(os.getpid(), sig_int)\n    time.sleep(1)\n    # Still running, test shall fail...\n    os.kill(os.getpid(), sig_kill)\n\nt = threading.Thread(target=interrupt, daemon=True)\nt.start()\n\n@crochet.%s\ndef wait():\n    return Deferred()\n\ntry:\n    wait()\nexcept KeyboardInterrupt:\n    sys.exit(23)\n" % (self.DECORATOR_CALL,)
    kw = {'cwd': crochet_directory}
    if platform.type.startswith('win'):
        kw['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen([sys.executable, '-c', program], **kw)
    self.assertEqual(process.wait(), 23)