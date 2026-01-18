import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_scheduler_hints(self):
    """
        Test that setting scheduler_hints will include them in POST request
        """
    scheduler_hints = {'group': self.getUniqueString('group')}
    fake_server = fakes.make_fake_server('1234', '', 'BUILD')
    fake_server['scheduler_hints'] = scheduler_hints
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': fake_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'name': 'server-name', 'networks': 'auto'}, 'OS-SCH-HNT:scheduler_hints': scheduler_hints})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': fake_server})])
    self.cloud.create_server(name='server-name', image=dict(id='image-id'), flavor=dict(id='flavor-id'), scheduler_hints=scheduler_hints, wait=False)
    self.assert_calls()