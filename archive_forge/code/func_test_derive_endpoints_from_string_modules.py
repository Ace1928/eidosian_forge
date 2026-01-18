import io
from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import worker
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
def test_derive_endpoints_from_string_modules(self):
    endpoints = worker.Worker._derive_endpoints(['taskflow.tests.utils'])
    assert any((e.name == self.task_name for e in endpoints))