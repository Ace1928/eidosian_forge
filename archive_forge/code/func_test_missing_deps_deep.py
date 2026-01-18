import collections
import contextlib
import functools
import threading
import futurist
import testtools
import taskflow.engines
from taskflow.engines.action_engine import engine as eng
from taskflow.engines.worker_based import engine as w_eng
from taskflow.engines.worker_based import worker as wkr
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow.persistence import models
from taskflow import states
from taskflow import task
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.types import graph as gr
from taskflow.utils import eventlet_utils as eu
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils as tu
def test_missing_deps_deep(self):
    flow = gf.Flow('missing-many').add(utils.TaskOneReturn(name='task1', requires=['a', 'b', 'c']), utils.TaskMultiArgOneReturn(name='task2', rebind=['e', 'f', 'g']))
    engine = self._make_engine(flow)
    engine.compile()
    engine.prepare()
    self.assertRaises(exc.MissingDependencies, engine.validate)
    c_e = None
    try:
        engine.validate()
    except exc.MissingDependencies as e:
        c_e = e
    self.assertIsNotNone(c_e)
    self.assertIsNotNone(c_e.cause)