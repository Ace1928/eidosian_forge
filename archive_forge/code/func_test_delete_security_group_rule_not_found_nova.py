import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_security_group_rule_not_found_nova(self):
    self.has_neutron = False
    self.cloud.secgroup_source = 'nova'
    self.register_uris([dict(method='GET', uri='{endpoint}/os-security-groups'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_groups': [nova_grp_dict]})])
    r = self.cloud.delete_security_group('doesNotExist')
    self.assertFalse(r)
    self.assert_calls()