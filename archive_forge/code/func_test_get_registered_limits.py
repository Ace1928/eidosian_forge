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
def test_get_registered_limits(self):
    fake_endpoint = endpoint.Endpoint()
    fake_endpoint.service_id = 'service_id'
    fake_endpoint.region_id = 'region_id'
    self.mock_conn.get_endpoint.return_value = fake_endpoint
    empty_iterator = iter([])
    a = registered_limit.RegisteredLimit()
    a.resource_name = 'a'
    a.default_limit = 1
    a_iterator = iter([a])
    c = registered_limit.RegisteredLimit()
    c.resource_name = 'c'
    c.default_limit = 2
    c_iterator = iter([c])
    self.mock_conn.registered_limits.side_effect = [a_iterator, empty_iterator, c_iterator]
    utils = limit._EnforcerUtils()
    limits = utils.get_registered_limits(['a', 'b', 'c'])
    self.assertEqual([('a', 1), ('b', 0), ('c', 2)], limits)