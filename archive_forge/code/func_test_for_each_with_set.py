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
def test_for_each_with_set(self):
    collection = set([3, 2, 5])
    retry1 = retry.ForEach(collection, 'r1', provides='x')
    flow = lf.Flow('flow-1', retry1).add(utils.FailingTaskWithOneArg('t1'))
    engine = self._make_engine(flow)
    with utils.CaptureListener(engine) as capturer:
        self.assertRaisesRegex(RuntimeError, '^Woot', engine.run)
    expected = ['flow-1.f RUNNING', 'r1.r RUNNING', 'r1.r SUCCESS(2)', 't1.t RUNNING', 't1.t FAILURE(Failure: RuntimeError: Woot with 2)', 't1.t REVERTING', 't1.t REVERTED(None)', 'r1.r RETRYING', 't1.t PENDING', 'r1.r RUNNING', 'r1.r SUCCESS(3)', 't1.t RUNNING', 't1.t FAILURE(Failure: RuntimeError: Woot with 3)', 't1.t REVERTING', 't1.t REVERTED(None)', 'r1.r RETRYING', 't1.t PENDING', 'r1.r RUNNING', 'r1.r SUCCESS(5)', 't1.t RUNNING', 't1.t FAILURE(Failure: RuntimeError: Woot with 5)', 't1.t REVERTING', 't1.t REVERTED(None)', 'r1.r REVERTING', 'r1.r REVERTED(None)', 'flow-1.f REVERTED']
    self.assertCountEqual(capturer.values, expected)