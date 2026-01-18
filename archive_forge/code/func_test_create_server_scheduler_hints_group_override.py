import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_scheduler_hints_group_override(self):
    """
        Test that setting group in both scheduler_hints and group param prefers
        param
        """
    group_id_scheduler_hints = uuid.uuid4().hex
    group_id_param = uuid.uuid4().hex
    group_name = self.getUniqueString('server-group')
    policies = ['affinity']
    fake_group = fakes.make_fake_server_group(group_id_param, group_name, policies)
    scheduler_hints = {'group': group_id_scheduler_hints}
    group_scheduler_hints = {'group': group_id_param}
    fake_server = fakes.make_fake_server('1234', '', 'BUILD')
    fake_server['scheduler_hints'] = group_scheduler_hints
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-server-groups']), json={'server_groups': [fake_group]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': fake_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'name': 'server-name', 'networks': 'auto'}, 'OS-SCH-HNT:scheduler_hints': group_scheduler_hints})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': fake_server})])
    self.cloud.create_server(name='server-name', image=dict(id='image-id'), flavor=dict(id='flavor-id'), scheduler_hints=dict(scheduler_hints), group=group_name, wait=False)
    self.assert_calls()