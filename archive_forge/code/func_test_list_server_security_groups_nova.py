import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_server_security_groups_nova(self):
    self.has_neutron = False
    server = fakes.make_fake_server('1234', 'server-name', 'ACTIVE')
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', server['id']]), json=server), dict(method='GET', uri='{endpoint}/servers/{id}/os-security-groups'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=server['id']), json={'security_groups': [nova_grp_dict]})])
    groups = self.cloud.list_server_security_groups(server)
    self.assertEqual(groups[0]['rules'][0]['ip_range']['cidr'], nova_grp_dict['rules'][0]['ip_range']['cidr'])
    self.assert_calls()