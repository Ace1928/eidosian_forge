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
@mark.skipif(windows, reason='ipc not supported on Windows.')
def test_ipc_path_no_such_file_or_directory_message(self):
    """Display the ipc path in case of an ENOENT exception"""
    s = self.context.socket(zmq.PUB)
    self.sockets.append(s)
    invalid_path = '/foo/bar'
    with pytest.raises(zmq.ZMQError) as error:
        s.bind(f'ipc://{invalid_path}')
    assert error.value.errno == errno.ENOENT
    error_message = str(error.value)
    assert invalid_path in error_message
    assert 'no such file or directory' in error_message.lower()