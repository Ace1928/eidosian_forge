import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_snapshot_instances_detail(self, **kw):
    instances = {'snapshot_instances': [{'id': '1234', 'snapshot_id': '5679', 'created_at': 'fake', 'updated_at': 'fake', 'status': 'fake', 'share_id': 'fake', 'share_instance_id': 'fake', 'progress': 'fake', 'provider_location': 'fake'}]}
    return (200, {}, instances)