import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_long_arg_name(self):
    flow = utils.LongArgNameTask(requires='long_arg_name', provides='result')
    engine = self._make_engine(flow)
    engine.storage.inject({'long_arg_name': 1})
    engine.run()
    self.assertEqual({'long_arg_name': 1, 'result': 1}, engine.storage.fetch_all())