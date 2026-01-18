import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_security_group_rule_not_found(self):
    rule_id = 'doesNotExist'
    self.cloud.secgroup_source = 'neutron'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups']), json={'security_groups': [neutron_grp_dict]})])
    self.assertFalse(self.cloud.delete_security_group(rule_id))
    self.assert_calls()