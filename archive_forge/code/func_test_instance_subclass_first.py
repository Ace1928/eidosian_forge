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
def test_instance_subclass_first(self):
    self.context.term()

    class SubContext(zmq.Context):
        pass
    sctx = SubContext.instance()
    ctx = zmq.Context.instance()
    ctx.term()
    sctx.term()
    assert type(ctx) is zmq.Context
    assert type(sctx) is SubContext