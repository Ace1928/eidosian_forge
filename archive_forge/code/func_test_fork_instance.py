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
@mark.skipif(sys.platform.startswith('win'), reason='No fork on Windows')
def test_fork_instance(self):
    ctx = self.Context.instance()
    parent_ctx_id = id(ctx)
    r_fd, w_fd = os.pipe()
    reader = os.fdopen(r_fd, 'r')
    child_pid = os.fork()
    if child_pid == 0:
        ctx = self.Context.instance()
        writer = os.fdopen(w_fd, 'w')
        child_ctx_id = id(ctx)
        ctx.term()
        writer.write(str(child_ctx_id) + '\n')
        writer.flush()
        writer.close()
        os._exit(0)
    else:
        os.close(w_fd)
    child_id_s = reader.readline()
    reader.close()
    assert child_id_s
    assert int(child_id_s) != parent_ctx_id
    ctx.term()