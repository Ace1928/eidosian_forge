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
def test_first_runs_reactor(self):
    """
        With it first call, setup() runs the reactor in a thread.
        """
    reactor = FakeReactor()
    EventLoop(lambda: reactor, lambda f, *g: None).setup()
    reactor.started.wait(5)
    self.assertNotEqual(reactor.thread_id, None)
    self.assertNotEqual(reactor.thread_id, threading.current_thread().ident)
    self.assertFalse(reactor.installSignalHandlers)