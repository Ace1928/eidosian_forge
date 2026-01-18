import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_security_group_neutron(self):
    sg_id = neutron_grp_dict['id']
    self.cloud.secgroup_source = 'neutron'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups']), json={'security_groups': [neutron_grp_dict]}), dict(method='DELETE', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups', '%s' % sg_id]), status_code=200, json={})])
    self.assertTrue(self.cloud.delete_security_group('1'))
    self.assert_calls()