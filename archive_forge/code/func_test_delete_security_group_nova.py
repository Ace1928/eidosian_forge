import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_security_group_nova(self):
    self.cloud.secgroup_source = 'nova'
    self.has_neutron = False
    nova_return = [nova_grp_dict]
    self.register_uris([dict(method='GET', uri='{endpoint}/os-security-groups'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_groups': nova_return}), dict(method='DELETE', uri='{endpoint}/os-security-groups/2'.format(endpoint=fakes.COMPUTE_ENDPOINT))])
    self.cloud.delete_security_group('2')
    self.assert_calls()