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
def test_eventloop_api(self):
    """
        An EventLoop object configured with the real reactor and
        _shutdown.register is exposed via its public methods.
        """
    from twisted.python.log import startLoggingWithObserver
    from crochet import _shutdown
    self.assertIsInstance(_main, EventLoop)
    self.assertEqual(_main.setup, setup_crochet)
    self.assertEqual(_main.no_setup, no_setup)
    self.assertEqual(_main.run_in_reactor, run_in_reactor)
    self.assertEqual(_main.wait_for, wait_for)
    self.assertIdentical(_main._atexit_register, _shutdown.register)
    self.assertIdentical(_main._startLoggingWithObserver, startLoggingWithObserver)
    self.assertIdentical(_main._watchdog_thread, _shutdown._watchdog)