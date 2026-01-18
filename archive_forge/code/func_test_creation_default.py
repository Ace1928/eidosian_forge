from taskflow.engines.worker_based import engine
from taskflow.engines.worker_based import executor
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.utils import persistence_utils as pu
def test_creation_default(self):
    executor_mock, executor_inst_mock = self._patch_in_executor()
    eng = self._create_engine()
    expected_calls = [mock.call.executor_class(uuid=eng.storage.flow_uuid, url=None, exchange='default', topics=[], transport=None, transport_options=None, transition_timeout=mock.ANY, retry_options=None, worker_expiry=mock.ANY)]
    self.assertEqual(expected_calls, self.master_mock.mock_calls)