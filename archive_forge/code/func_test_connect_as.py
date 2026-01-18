from unittest import mock
import uuid
import testtools
from openstack import connection
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_connect_as(self):
    project_name = 'test_project'
    self.register_uris([self.get_keystone_v3_token(project_name=project_name), self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': []})])
    c2 = self.cloud.connect_as(project_name=project_name)
    self.assertEqual(c2.list_servers(), [])
    self.assert_calls()