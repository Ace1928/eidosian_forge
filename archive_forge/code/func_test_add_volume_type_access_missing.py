import testtools
from openstack import exceptions
from openstack.tests.unit import base
def test_add_volume_type_access_missing(self):
    volume_type = dict(id='voltype01', description='volume type description', name='name', is_public=False)
    project_001 = dict(volume_type_id='voltype01', name='name', project_id='prj01')
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['types']), json={'volume_types': [volume_type]})])
    with testtools.ExpectedException(exceptions.SDKException, 'VolumeType not found: MISSING'):
        self.cloud.add_volume_type_access('MISSING', project_001['project_id'])
    self.assert_calls()