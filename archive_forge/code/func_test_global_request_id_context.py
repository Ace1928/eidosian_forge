from unittest import mock
import uuid
import testtools
from openstack import connection
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_global_request_id_context(self):
    request_id = uuid.uuid4().hex
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': []}, validate=dict(headers={'X-Openstack-Request-Id': request_id}))])
    with self.cloud.global_request(request_id) as c2:
        self.assertEqual([], c2.list_servers())
    self.assert_calls()