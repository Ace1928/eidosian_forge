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
def test_bind_to_random_port(self):
    ctx = self.Context()
    s = ctx.socket(zmq.PUB)
    try:
        s.bind_to_random_port('tcp:*')
    except zmq.ZMQError as e:
        assert e.errno == zmq.EINVAL
    try:
        s.bind_to_random_port('rand://*')
    except zmq.ZMQError as e:
        assert e.errno == zmq.EPROTONOSUPPORT
    s.close()
    ctx.term()