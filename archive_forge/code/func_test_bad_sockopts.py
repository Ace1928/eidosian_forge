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
def test_bad_sockopts(self):
    """Test that appropriate errors are raised on bad socket options"""
    s = self.context.socket(zmq.PUB)
    self.sockets.append(s)
    s.setsockopt(zmq.LINGER, 0)
    self.assertRaisesErrno(zmq.EINVAL, s.setsockopt, 9999, 5)
    self.assertRaisesErrno(zmq.EINVAL, s.getsockopt, 9999)
    self.assertRaises(TypeError, s.setsockopt, 9999, b'5')
    self.assertRaisesErrno(zmq.EINVAL, s.setsockopt, zmq.SUBSCRIBE, b'hi')