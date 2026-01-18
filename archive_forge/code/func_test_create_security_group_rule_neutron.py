import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_security_group_rule_neutron(self):
    self.cloud.secgroup_source = 'neutron'
    args = dict(port_range_min=-1, port_range_max=40000, protocol='tcp', remote_ip_prefix='0.0.0.0/0', remote_group_id='456', remote_address_group_id='1234-5678', direction='egress', ethertype='IPv6')
    expected_args = copy.copy(args)
    expected_args['port_range_min'] = None
    expected_args['security_group_id'] = neutron_grp_dict['id']
    expected_new_rule = copy.copy(expected_args)
    expected_new_rule['id'] = '1234'
    expected_new_rule['tenant_id'] = None
    expected_new_rule['project_id'] = expected_new_rule['tenant_id']
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups']), json={'security_groups': [neutron_grp_dict]}), dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-group-rules']), json={'security_group_rule': expected_new_rule}, validate=dict(json={'security_group_rule': expected_args}))])
    new_rule = self.cloud.create_security_group_rule(secgroup_name_or_id=neutron_grp_dict['id'], **args).to_dict(original_names=True)
    new_rule.pop('created_at')
    new_rule.pop('description')
    new_rule.pop('location')
    new_rule.pop('name')
    new_rule.pop('revision_number')
    new_rule.pop('tags')
    new_rule.pop('updated_at')
    self.assertEqual(expected_new_rule, new_rule)
    self.assert_calls()