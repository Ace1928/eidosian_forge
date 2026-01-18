import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_security_group_by_id_nova(self):
    self.register_uris([dict(method='GET', uri='{endpoint}/os-security-groups/{id}'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=nova_grp_dict['id']), json={'security_group': nova_grp_dict})])
    self.cloud.secgroup_source = 'nova'
    self.has_neutron = False
    ret_sg = self.cloud.get_security_group_by_id(nova_grp_dict['id'])
    self.assertEqual(nova_grp_dict['id'], ret_sg['id'])
    self.assertEqual(nova_grp_dict['name'], ret_sg['name'])
    self.assert_calls()