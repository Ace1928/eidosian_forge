import futurist
import testtools
from taskflow.engines.action_engine import engine
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import process_executor
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
from taskflow.utils import persistence_utils as pu
def test_invalid_creation(self):
    self.assertRaises(ValueError, self._create_engine, executor='crap')
    self.assertRaises(TypeError, self._create_engine, executor=2)
    self.assertRaises(TypeError, self._create_engine, executor=object())