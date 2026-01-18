from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_creation_with_endpoints(self):
    s = self.server(endpoints=self.endpoints)
    master_mock_calls = [mock.call.Proxy(self.server_topic, self.server_exchange, type_handlers=mock.ANY, url=self.broker_url, transport=mock.ANY, transport_options=mock.ANY, retry_options=mock.ANY)]
    self.master_mock.assert_has_calls(master_mock_calls)
    self.assertEqual(len(self.endpoints), len(s._endpoints))