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
def test_ipc_path_max_length_msg(self):
    if zmq.IPC_PATH_MAX_LEN == 0:
        raise SkipTest('IPC_PATH_MAX_LEN undefined')
    s = self.context.socket(zmq.PUB)
    self.sockets.append(s)
    try:
        s.bind('ipc://{}'.format('a' * (zmq.IPC_PATH_MAX_LEN + 1)))
    except zmq.ZMQError as e:
        assert str(zmq.IPC_PATH_MAX_LEN) in e.strerror