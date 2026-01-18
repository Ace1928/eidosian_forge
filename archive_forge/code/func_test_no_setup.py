from __future__ import absolute_import
import threading
import warnings
import subprocess
import sys
from unittest import SkipTest, TestCase
import twisted
from twisted.python.log import PythonLoggingObserver
from twisted.python import log
from twisted.python.runtime import platform
from twisted.internet.task import Clock
from .._eventloop import EventLoop, ThreadLogObserver, _store
from ..tests import crochet_directory
import sys
import crochet
import sys
from logging import StreamHandler, Formatter, getLogger, DEBUG
import crochet
from twisted.python import log
from twisted.logger import Logger
import time
def test_no_setup(self):
    """
        If called first, no_setup() makes subsequent calls to setup() do
        nothing.
        """
    observers = []
    atexit = []
    thread = FakeThread()
    reactor = FakeReactor()
    loop = EventLoop(lambda: reactor, lambda f, *arg: atexit.append(f), lambda observer, *a, **kw: observers.append(observer), watchdog_thread=thread)
    loop.no_setup()
    loop.setup()
    self.assertFalse(observers)
    self.assertFalse(atexit)
    self.assertFalse(reactor.runs)
    self.assertFalse(thread.started)