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
def test_get_project_limits(self):
    fake_endpoint = endpoint.Endpoint()
    fake_endpoint.service_id = 'service_id'
    fake_endpoint.region_id = 'region_id'
    self.mock_conn.get_endpoint.return_value = fake_endpoint
    project_id = uuid.uuid4().hex
    empty_iterator = iter([])
    a = klimit.Limit()
    a.resource_name = 'a'
    a.resource_limit = 1
    a_iterator = iter([a])
    self.mock_conn.limits.side_effect = [a_iterator, empty_iterator, empty_iterator, empty_iterator]
    b = registered_limit.RegisteredLimit()
    b.resource_name = 'b'
    b.default_limit = 2
    b_iterator = iter([b])
    self.mock_conn.registered_limits.side_effect = [b_iterator, empty_iterator, empty_iterator]
    utils = limit._EnforcerUtils()
    limits = utils.get_project_limits(project_id, ['a', 'b'])
    self.assertEqual([('a', 1), ('b', 2)], limits)
    limits = utils.get_project_limits(project_id, ['c', 'd'])
    self.assertEqual([('c', 0), ('d', 0)], limits)