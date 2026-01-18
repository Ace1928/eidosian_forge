import json
from oslo_policy import fixture
from oslo_policy import policy as oslo_policy
from oslo_policy.tests import base as test_base
def test_enforce_http_true(self):
    self.assertTrue(self._test_enforce_http(True))