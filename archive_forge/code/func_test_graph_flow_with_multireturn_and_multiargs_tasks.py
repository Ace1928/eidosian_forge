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
def test_graph_flow_with_multireturn_and_multiargs_tasks(self):
    flow = gf.Flow('g-3-multi').add(utils.TaskMultiArgOneReturn(name='task1', rebind=['a', 'b', 'y'], provides='z'), utils.TaskMultiReturn(name='task2', provides=['a', 'b', 'c']), utils.TaskMultiArgOneReturn(name='task3', rebind=['c', 'b', 'x'], provides='y'))
    engine = self._make_engine(flow)
    engine.storage.inject({'x': 30})
    engine.run()
    self.assertEqual({'a': 1, 'b': 3, 'c': 5, 'x': 30, 'y': 38, 'z': 42}, engine.storage.fetch_all())