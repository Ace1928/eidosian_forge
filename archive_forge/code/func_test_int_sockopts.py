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
def test_int_sockopts(self):
    """test integer sockopts"""
    v = zmq.zmq_version_info()
    if v < (3, 0):
        default_hwm = 0
    else:
        default_hwm = 1000
    p, s = self.create_bound_pair(zmq.PUB, zmq.SUB)
    p.setsockopt(zmq.LINGER, 0)
    assert p.getsockopt(zmq.LINGER) == 0
    p.setsockopt(zmq.LINGER, -1)
    assert p.getsockopt(zmq.LINGER) == -1
    assert p.hwm == default_hwm
    p.hwm = 11
    assert p.hwm == 11
    assert p.getsockopt(zmq.EVENTS) == zmq.POLLOUT
    self.assertRaisesErrno(zmq.EINVAL, p.setsockopt, zmq.EVENTS, 2 ** 7 - 1)
    assert p.getsockopt(zmq.TYPE) == p.socket_type
    assert p.getsockopt(zmq.TYPE) == zmq.PUB
    assert s.getsockopt(zmq.TYPE) == s.socket_type
    assert s.getsockopt(zmq.TYPE) == zmq.SUB
    errors = []
    backref = {}
    constants = zmq.constants
    for name in constants.__all__:
        value = getattr(constants, name)
        if isinstance(value, int):
            backref[value] = name
    for opt in zmq.constants.SocketOption:
        if opt._opt_type not in {zmq.constants._OptType.int, zmq.constants._OptType.int64}:
            continue
        if opt.name.startswith(('HWM', 'ROUTER', 'XPUB', 'TCP', 'FAIL', 'REQ_', 'CURVE_', 'PROBE_ROUTER', 'IPC_FILTER', 'GSSAPI', 'STREAM_', 'VMCI_BUFFER_SIZE', 'VMCI_BUFFER_MIN_SIZE', 'VMCI_BUFFER_MAX_SIZE', 'VMCI_CONNECT_TIMEOUT', 'BLOCKY', 'IN_BATCH_SIZE', 'OUT_BATCH_SIZE', 'WSS_TRUST_SYSTEM', 'ONLY_FIRST_SUBSCRIBE', 'PRIORITY', 'RECONNECT_STOP', 'NORM_', 'ROUTER_', 'BUSY_POLL', 'XSUB_VERBOSE_', 'TOPICS_')):
            continue
        try:
            n = p.getsockopt(opt)
        except zmq.ZMQError as e:
            errors.append(f'getsockopt({opt!r}) raised {e}.')
        else:
            if n > 2 ** 31:
                errors.append(f'getsockopt({opt!r}) returned a ridiculous value. It is probably the wrong type.')
    if errors:
        self.fail('\n'.join([''] + errors))