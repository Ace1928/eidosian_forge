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
def append_in_thread(a_list, f, *args, **kwargs):
    """
    Call a function in a thread, append its result to the given list.

    Only return once the thread has actually started.

    Will return a threading.Event that will be set when the action is done.
    """
    started = threading.Event()
    done = threading.Event()

    def go():
        started.set()
        try:
            result = f(*args, **kwargs)
        except Exception as e:
            a_list.extend([False, e])
        else:
            a_list.extend([True, result])
        done.set()
    threading.Thread(target=go).start()
    started.wait()
    return done