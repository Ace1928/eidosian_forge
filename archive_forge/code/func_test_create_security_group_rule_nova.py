import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_security_group_rule_nova(self):
    self.has_neutron = False
    self.cloud.secgroup_source = 'nova'
    nova_return = [nova_grp_dict]
    new_rule = fakes.make_fake_nova_security_group_rule(id='xyz', from_port=1, to_port=2000, ip_protocol='tcp', cidr='1.2.3.4/32')
    self.register_uris([dict(method='GET', uri='{endpoint}/os-security-groups'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_groups': nova_return}), dict(method='POST', uri='{endpoint}/os-security-group-rules'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_group_rule': new_rule}, validate=dict(json={'security_group_rule': {'from_port': 1, 'ip_protocol': 'tcp', 'to_port': 2000, 'parent_group_id': '2', 'cidr': '1.2.3.4/32', 'group_id': '123'}}))])
    self.cloud.create_security_group_rule('2', port_range_min=1, port_range_max=2000, protocol='tcp', remote_ip_prefix='1.2.3.4/32', remote_group_id='123')
    self.assert_calls()