from taskflow.engines.worker_based import engine
from taskflow.engines.worker_based import executor
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.utils import persistence_utils as pu
def test_creation_custom(self):
    executor_mock, executor_inst_mock = self._patch_in_executor()
    topics = ['test-topic1', 'test-topic2']
    exchange = 'test-exchange'
    broker_url = 'test-url'
    eng = self._create_engine(url=broker_url, exchange=exchange, transport='memory', transport_options={}, transition_timeout=200, topics=topics, retry_options={}, worker_expiry=1)
    expected_calls = [mock.call.executor_class(uuid=eng.storage.flow_uuid, url=broker_url, exchange=exchange, topics=topics, transport='memory', transport_options={}, transition_timeout=200, retry_options={}, worker_expiry=1)]
    self.assertEqual(expected_calls, self.master_mock.mock_calls)