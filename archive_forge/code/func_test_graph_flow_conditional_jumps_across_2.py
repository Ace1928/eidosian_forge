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
def test_graph_flow_conditional_jumps_across_2(self):
    histories = []

    def should_go(history):
        histories.append(history)
        return False
    task1 = utils.ProgressingTask(name='task1')
    task2 = utils.ProgressingTask(name='task2')
    task3 = utils.ProgressingTask(name='task3')
    task4 = utils.ProgressingTask(name='task4')
    subflow = lf.Flow('more-work')
    subsub_flow = lf.Flow('more-more-work')
    subsub_flow.add(task3, task4)
    subflow.add(subsub_flow)
    flow = gf.Flow('main-work')
    flow.add(task1, task2)
    flow.link(task1, task2)
    flow.add(subflow)
    flow.link(task2, subflow, decider=should_go)
    engine = self._make_engine(flow)
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        engine.run()
    expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t SUCCESS(5)', 'task3.t IGNORE', 'task4.t IGNORE']
    self.assertEqual(expected, capturer.values)
    self.assertEqual(1, len(histories))
    self.assertIn('task2', histories[0])