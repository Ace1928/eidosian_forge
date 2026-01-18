import copy
import errno
import json
import os
import platform
import socket
import sys
import time
import warnings
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, SkipTest, have_gevent, skip_pypy
@mark.skipif(pypy and on_ci or sys.maxsize < 2 ** 32 or windows, reason='only run on 64b and not on CI.')
@mark.large
def test_large_send(self):
    c = os.urandom(1)
    N = 2 ** 31 + 1
    try:
        buf = c * N
    except MemoryError as e:
        raise SkipTest('Not enough memory: %s' % e)
    a, b = self.create_bound_pair()
    try:
        a.send(buf, copy=False)
        rcvd = b.recv(copy=False)
    except MemoryError as e:
        raise SkipTest('Not enough memory: %s' % e)
    byte = ord(c)
    view = memoryview(rcvd)
    assert len(view) == N
    assert view[0] == byte
    assert view[-1] == byte