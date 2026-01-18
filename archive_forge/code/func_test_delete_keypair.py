import fixtures
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_keypair(self):
    self.register_uris([dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['os-keypairs', self.keyname]), status_code=202)])
    self.assertTrue(self.cloud.delete_keypair(self.keyname))
    self.assert_calls()