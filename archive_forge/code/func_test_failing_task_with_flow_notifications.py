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
def test_failing_task_with_flow_notifications(self):
    values = []
    flow = utils.FailingTask('fail')
    engine = self._make_engine(flow)
    expected = ['fail.f RUNNING', 'fail.t RUNNING', 'fail.t FAILURE(Failure: RuntimeError: Woot!)', 'fail.t REVERTING', 'fail.t REVERTED(None)', 'fail.f REVERTED']
    with utils.CaptureListener(engine, values=values) as capturer:
        self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
    self.assertEqual(expected, capturer.values)
    self.assertEqual(states.REVERTED, engine.storage.get_flow_state())
    with utils.CaptureListener(engine, values=values) as capturer:
        self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
    now_expected = list(expected)
    now_expected.extend(['fail.t PENDING', 'fail.f PENDING'])
    now_expected.extend(expected)
    self.assertEqual(now_expected, values)
    self.assertEqual(states.REVERTED, engine.storage.get_flow_state())