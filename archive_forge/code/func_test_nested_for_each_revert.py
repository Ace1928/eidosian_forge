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
def test_nested_for_each_revert(self):
    collection = [3, 2, 3, 5]
    retry1 = retry.ForEach(collection, 'r1', provides='x')
    flow = lf.Flow('flow-1').add(utils.ProgressingTask('task1'), lf.Flow('flow-2', retry1).add(utils.FailingTaskWithOneArg('task2')))
    engine = self._make_engine(flow)
    with utils.CaptureListener(engine) as capturer:
        self.assertRaisesRegex(RuntimeError, '^Woot', engine.run)
    expected = ['flow-1.f RUNNING', 'task1.t RUNNING', 'task1.t SUCCESS(5)', 'r1.r RUNNING', 'r1.r SUCCESS(3)', 'task2.t RUNNING', 'task2.t FAILURE(Failure: RuntimeError: Woot with 3)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'r1.r RETRYING', 'task2.t PENDING', 'r1.r RUNNING', 'r1.r SUCCESS(2)', 'task2.t RUNNING', 'task2.t FAILURE(Failure: RuntimeError: Woot with 2)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'r1.r RETRYING', 'task2.t PENDING', 'r1.r RUNNING', 'r1.r SUCCESS(3)', 'task2.t RUNNING', 'task2.t FAILURE(Failure: RuntimeError: Woot with 3)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'r1.r RETRYING', 'task2.t PENDING', 'r1.r RUNNING', 'r1.r SUCCESS(5)', 'task2.t RUNNING', 'task2.t FAILURE(Failure: RuntimeError: Woot with 5)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'r1.r REVERTING', 'r1.r REVERTED(None)', 'flow-1.f REVERTED']
    self.assertEqual(expected, capturer.values)