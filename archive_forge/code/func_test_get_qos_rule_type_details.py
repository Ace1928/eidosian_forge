from openstack import exceptions
from openstack.network.v2 import qos_rule_type
from openstack.tests.unit import base
def test_get_qos_rule_type_details(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': [self.qos_extension, self.qos_rule_type_details_extension]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': [self.qos_extension, self.qos_rule_type_details_extension]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'rule-types', self.rule_type_name]), json={'rule_type': self.mock_rule_type_details})])
    self._compare_rule_types(self.mock_rule_type_details, self.cloud.get_qos_rule_type_details(self.rule_type_name))
    self.assert_calls()