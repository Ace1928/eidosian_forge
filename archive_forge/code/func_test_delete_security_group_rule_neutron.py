import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_security_group_rule_neutron(self):
    rule_id = 'xyz'
    self.cloud.secgroup_source = 'neutron'
    self.register_uris([dict(method='DELETE', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-group-rules', '%s' % rule_id]), json={})])
    self.assertTrue(self.cloud.delete_security_group_rule(rule_id))
    self.assert_calls()