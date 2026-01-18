import copy
import gc
import os
import sys
import time
from queue import Queue
from threading import Event, Thread
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest
def test_destroy_linger(self):
    """Context.destroy should set linger on closing sockets"""
    req, rep = self.create_bound_pair(zmq.REQ, zmq.REP)
    req.send(b'hi')
    time.sleep(0.01)
    self.context.destroy(linger=0)
    time.sleep(0.01)
    for s in (req, rep):
        assert s.closed