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
def test_bind_connect_addr_error(self):
    with self.socket(zmq.PUSH) as s:
        url = 'tcp://1.2.3.4.5:1234567'
        with pytest.raises(zmq.ZMQError) as exc:
            s.bind(url)
        assert url in str(exc.value)
        url = 'noproc://no/such/file'
        with pytest.raises(zmq.ZMQError) as exc:
            s.connect(url)
        assert url in str(exc.value)