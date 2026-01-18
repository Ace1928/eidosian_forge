import fixtures
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_keypairs_empty_filters(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-keypairs']), json={'keypairs': [{'keypair': self.key}]})])
    keypairs = self.cloud.list_keypairs(filters=None)
    self.assertEqual(len(keypairs), 1)
    self.assertEqual(keypairs[0].name, self.key['name'])
    self.assert_calls()