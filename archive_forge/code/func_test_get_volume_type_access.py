import testtools
from openstack import exceptions
from openstack.tests.unit import base
def test_get_volume_type_access(self):
    volume_type = dict(id='voltype01', description='volume type description', name='name', is_public=False)
    volume_type_access = [dict(volume_type_id='voltype01', name='name', project_id='prj01'), dict(volume_type_id='voltype01', name='name', project_id='prj02')]
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['types']), json={'volume_types': [volume_type]}), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['types', volume_type['id'], 'os-volume-type-access']), json={'volume_type_access': volume_type_access})])
    self.assertEqual(len(self.cloud.get_volume_type_access(volume_type['name'])), 2)
    self.assert_calls()