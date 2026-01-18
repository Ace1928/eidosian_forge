import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
def test_enforcer_consolidates_context_attributes_with_creds(self):
    request_context = context.RequestContext()
    expected_creds = request_context.to_policy_values()
    creds = self.enforcer._map_context_attributes_into_creds(request_context)
    for k, v in expected_creds.items():
        self.assertEqual(expected_creds[k], creds[k])