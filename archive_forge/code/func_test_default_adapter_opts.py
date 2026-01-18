import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_default_adapter_opts(self):
    """Adapter opts are registered, but all defaulting in conf."""
    conn = self._get_conn()
    server_id = str(uuid.uuid4())
    server_name = self.getUniqueString('name')
    fake_server = fakes.make_fake_server(server_id, server_name)
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': [fake_server]})])
    adap = conn.compute
    self.assertIsNone(adap.region_name)
    self.assertEqual('compute', adap.service_type)
    self.assertEqual('public', adap.interface)
    self.assertIsNone(adap.endpoint_override)
    s = next(adap.servers())
    self.assertEqual(s.id, server_id)
    self.assertEqual(s.name, server_name)
    self.assert_calls()