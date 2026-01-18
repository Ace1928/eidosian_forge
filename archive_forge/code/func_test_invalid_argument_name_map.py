import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_invalid_argument_name_map(self):
    flow = utils.TaskMultiArg(rebind={'z': 'b'})
    engine = self._make_engine(flow)
    engine.storage.inject({'a': 1, 'y': 4, 'c': 9, 'x': 17})
    self.assertRaises(exc.MissingDependencies, engine.run)