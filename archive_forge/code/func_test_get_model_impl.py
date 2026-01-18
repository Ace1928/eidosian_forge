from unittest import mock
import uuid
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import limit as klimit
from openstack.identity.v3 import registered_limit
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
def test_get_model_impl(self):
    json = mock.MagicMock()
    limit._SDK_CONNECTION.get.return_value = json
    json.json.return_value = {'model': {'name': 'flat'}}
    enforcer = limit.Enforcer(self._get_usage_for_project)
    flat_impl = enforcer._get_model_impl(self._get_usage_for_project)
    self.assertIsInstance(flat_impl, limit._FlatEnforcer)
    json.json.return_value = {'model': {'name': 'strict-two-level'}}
    flat_impl = enforcer._get_model_impl(self._get_usage_for_project)
    self.assertIsInstance(flat_impl, limit._StrictTwoLevelEnforcer)
    json.json.return_value = {'model': {'name': 'foo'}}
    e = self.assertRaises(ValueError, enforcer._get_model_impl, self._get_usage_for_project)
    self.assertEqual('enforcement model foo is not supported', str(e))