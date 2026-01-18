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
def test_instance_subclass_second(self):
    self.context.term()

    class SubContextInherit(zmq.Context):
        pass

    class SubContextNoInherit(zmq.Context):
        _instance = None
    ctx = zmq.Context.instance()
    sctx = SubContextInherit.instance()
    sctx2 = SubContextNoInherit.instance()
    ctx.term()
    sctx.term()
    sctx2.term()
    assert type(ctx) is zmq.Context
    assert type(sctx) is zmq.Context
    assert type(sctx2) is SubContextNoInherit