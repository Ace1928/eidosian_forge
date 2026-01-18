import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.utils import eventlet_utils as eu
def test_default_times_retry(self):
    flow = lf.Flow('flow-1', retry.Times(3, 'r1')).add(utils.ProgressingTask('t1'), utils.FailingTask('t2'))
    engine = self._make_engine(flow)
    with utils.CaptureListener(engine) as capturer:
        self.assertRaisesRegex(RuntimeError, '^Woot', engine.run)
    expected = ['flow-1.f RUNNING', 'r1.r RUNNING', 'r1.r SUCCESS(1)', 't1.t RUNNING', 't1.t SUCCESS(5)', 't2.t RUNNING', 't2.t FAILURE(Failure: RuntimeError: Woot!)', 't2.t REVERTING', 't2.t REVERTED(None)', 't1.t REVERTING', 't1.t REVERTED(None)', 'r1.r RETRYING', 't1.t PENDING', 't2.t PENDING', 'r1.r RUNNING', 'r1.r SUCCESS(2)', 't1.t RUNNING', 't1.t SUCCESS(5)', 't2.t RUNNING', 't2.t FAILURE(Failure: RuntimeError: Woot!)', 't2.t REVERTING', 't2.t REVERTED(None)', 't1.t REVERTING', 't1.t REVERTED(None)', 'r1.r RETRYING', 't1.t PENDING', 't2.t PENDING', 'r1.r RUNNING', 'r1.r SUCCESS(3)', 't1.t RUNNING', 't1.t SUCCESS(5)', 't2.t RUNNING', 't2.t FAILURE(Failure: RuntimeError: Woot!)', 't2.t REVERTING', 't2.t REVERTED(None)', 't1.t REVERTING', 't1.t REVERTED(None)', 'r1.r REVERTING', 'r1.r REVERTED(None)', 'flow-1.f REVERTED']
    self.assertEqual(expected, capturer.values)