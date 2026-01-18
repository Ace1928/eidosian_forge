import io
from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import worker
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
def test_derive_endpoints_from_string_tasks(self):
    endpoints = worker.Worker._derive_endpoints(['taskflow.tests.utils:DummyTask'])
    self.assertEqual(1, len(endpoints))
    self.assertIsInstance(endpoints[0], endpoint.Endpoint)
    self.assertEqual(self.task_name, endpoints[0].name)