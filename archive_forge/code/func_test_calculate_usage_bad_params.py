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
def test_calculate_usage_bad_params(self):
    enforcer = limit.Enforcer(mock.MagicMock())
    self.assertRaises(ValueError, enforcer.calculate_usage, 123, ['foo'])
    self.assertRaises(ValueError, enforcer.calculate_usage, 'project', [])
    self.assertRaises(ValueError, enforcer.calculate_usage, 'project', 123)
    self.assertRaises(ValueError, enforcer.calculate_usage, 'project', ['a', 123, 'b'])