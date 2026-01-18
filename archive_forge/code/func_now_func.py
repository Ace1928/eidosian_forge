import contextlib
import functools
import threading
import time
from unittest import mock
import eventlet
from eventlet.green import threading as green_threading
import testscenarios
import futurist
from futurist import periodics
from futurist.tests import base
def now_func():
    if len(nows) == 1:
        ev.set()
        return last_now
    return nows.pop()