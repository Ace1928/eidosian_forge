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
def test_nested_provides_graph_retried_correctly(self):
    flow = gf.Flow('test').add(utils.ProgressingTask('a', requires=['x']), lf.Flow('test2', retry=retry.Times(2)).add(utils.ProgressingTask('b', provides='x'), utils.ProgressingTask('c')))
    engine = self._make_engine(flow)
    engine.compile()
    engine.prepare()
    engine.storage.save('test2_retry', 1)
    engine.storage.save('b', 11)
    fail = failure.Failure.from_exception(RuntimeError('Woot!'))
    engine.storage.save('c', fail, st.FAILURE)
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        engine.run()
    expected = ['c.t REVERTING', 'c.t REVERTED(None)', 'b.t REVERTING', 'b.t REVERTED(None)']
    self.assertCountEqual(capturer.values[:4], expected)
    expected = ['test2_retry.r RETRYING', 'b.t PENDING', 'c.t PENDING', 'test2_retry.r RUNNING', 'test2_retry.r SUCCESS(2)', 'b.t RUNNING', 'b.t SUCCESS(5)', 'a.t RUNNING', 'c.t RUNNING', 'a.t SUCCESS(5)', 'c.t SUCCESS(5)']
    self.assertCountEqual(expected, capturer.values[4:])
    self.assertEqual(st.SUCCESS, engine.storage.get_flow_state())