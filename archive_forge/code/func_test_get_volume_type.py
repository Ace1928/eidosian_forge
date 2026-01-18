import testtools
from openstack import exceptions
from openstack.tests.unit import base
def test_get_volume_type(self):
    volume_type = dict(id='voltype01', description='volume type description', name='name', is_public=False)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['types']), json={'volume_types': [volume_type]})])
    volume_type_got = self.cloud.get_volume_type(volume_type['name'])
    self.assertEqual(volume_type_got.id, volume_type['id'])