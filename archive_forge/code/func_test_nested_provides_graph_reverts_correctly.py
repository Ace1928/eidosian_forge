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
def test_nested_provides_graph_reverts_correctly(self):
    flow = gf.Flow('test').add(utils.ProgressingTask('a', requires=['x']), lf.Flow('test2', retry=retry.Times(2)).add(utils.ProgressingTask('b', provides='x'), utils.FailingTask('c')))
    engine = self._make_engine(flow)
    engine.compile()
    engine.prepare()
    engine.storage.save('test2_retry', 1)
    engine.storage.save('b', 11)
    engine.storage.save('a', 10)
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        self.assertRaisesRegex(RuntimeError, '^Woot', engine.run)
    expected = ['c.t RUNNING', 'c.t FAILURE(Failure: RuntimeError: Woot!)', 'a.t REVERTING', 'c.t REVERTING', 'a.t REVERTED(None)', 'c.t REVERTED(None)', 'b.t REVERTING', 'b.t REVERTED(None)']
    self.assertCountEqual(capturer.values[:8], expected)
    self.assertIsSuperAndSubsequence(capturer.values[8:], ['b.t RUNNING', 'c.t FAILURE(Failure: RuntimeError: Woot!)', 'b.t REVERTED(None)'])
    self.assertEqual(st.REVERTED, engine.storage.get_flow_state())