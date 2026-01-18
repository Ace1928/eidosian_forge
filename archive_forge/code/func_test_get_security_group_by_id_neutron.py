import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_security_group_by_id_neutron(self):
    self.cloud.secgroup_source = 'neutron'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups', neutron_grp_dict['id']]), json={'security_group': neutron_grp_dict})])
    ret_sg = self.cloud.get_security_group_by_id(neutron_grp_dict['id'])
    self.assertEqual(neutron_grp_dict['id'], ret_sg['id'])
    self.assertEqual(neutron_grp_dict['name'], ret_sg['name'])
    self.assertEqual(neutron_grp_dict['description'], ret_sg['description'])
    self.assertEqual(neutron_grp_dict['stateful'], ret_sg['stateful'])
    self.assert_calls()