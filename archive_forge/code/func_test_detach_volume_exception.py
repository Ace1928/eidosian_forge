import testtools
from openstack.block_storage.v3 import volume
from openstack.cloud import meta
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_detach_volume_exception(self):
    server = dict(id='server001')
    volume = dict(id='volume001', attachments=[{'server_id': 'server001', 'device': 'device001'}])
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', server['id']]), json={'server': server}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['servers', server['id'], 'os-volume_attachments', volume['id']]), status_code=404)])
    with testtools.ExpectedException(exceptions.NotFoundException):
        self.cloud.detach_volume(server, volume, wait=False)
    self.assert_calls()