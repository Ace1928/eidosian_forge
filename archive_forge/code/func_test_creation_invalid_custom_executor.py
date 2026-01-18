from taskflow.engines.worker_based import engine
from taskflow.engines.worker_based import executor
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.utils import persistence_utils as pu
def test_creation_invalid_custom_executor(self):
    self.assertRaises(TypeError, self._create_engine, executor=2)
    self.assertRaises(TypeError, self._create_engine, executor='blah')