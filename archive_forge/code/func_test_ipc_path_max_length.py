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
def test_ipc_path_max_length(self):
    """IPC_PATH_MAX_LEN is a sensible value"""
    if zmq.IPC_PATH_MAX_LEN == 0:
        raise SkipTest('IPC_PATH_MAX_LEN undefined')
    msg = 'Surprising value for IPC_PATH_MAX_LEN: %s' % zmq.IPC_PATH_MAX_LEN
    assert zmq.IPC_PATH_MAX_LEN > 30, msg
    assert zmq.IPC_PATH_MAX_LEN < 1025, msg