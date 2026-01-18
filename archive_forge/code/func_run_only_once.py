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
@periodics.periodic(0.5)
def run_only_once():
    raise periodics.NeverAgain('No need to run again !!')