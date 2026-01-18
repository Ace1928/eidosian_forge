import testtools
from openstack import exceptions
from openstack.tests.unit import base
def test_list_volume_types(self):
    volume_type = dict(id='voltype01', description='volume type description', name='name', is_public=False)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['types']), json={'volume_types': [volume_type]})])
    self.assertTrue(self.cloud.list_volume_types())
    self.assert_calls()