import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_all_arguments_mapping(self):
    flow = utils.TaskMultiArgOneReturn(provides='result', rebind=['a', 'b', 'c'])
    engine = self._make_engine(flow)
    engine.storage.inject({'a': 1, 'b': 2, 'c': 3, 'x': 4, 'y': 5, 'z': 6})
    engine.run()
    self.assertEqual({'a': 1, 'b': 2, 'c': 3, 'x': 4, 'y': 5, 'z': 6, 'result': 6}, engine.storage.fetch_all())