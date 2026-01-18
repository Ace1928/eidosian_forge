import testtools
from openstack.block_storage.v3 import volume
from openstack.cloud import meta
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_attach_volume(self):
    server = dict(id='server001')
    vol = {'id': 'volume001', 'status': 'available', 'name': '', 'attachments': []}
    volume = meta.obj_to_munch(fakes.FakeVolume(**vol))
    rattach = {'server_id': server['id'], 'device': 'device001', 'volumeId': volume['id'], 'id': 'attachmentId'}
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers', server['id'], 'os-volume_attachments']), json={'volumeAttachment': rattach}, validate=dict(json={'volumeAttachment': {'volumeId': vol['id']}}))])
    ret = self.cloud.attach_volume(server, volume, wait=False)
    self._compare_volume_attachments(rattach, ret)
    self.assert_calls()