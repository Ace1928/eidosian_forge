from openstack import exceptions
from openstack.network.v2 import qos_rule_type
from openstack.tests.unit import base
def test_list_qos_rule_types(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': [self.qos_extension]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'rule-types']), json={'rule_types': self.mock_rule_types})])
    rule_types = self.cloud.list_qos_rule_types()
    for a, b in zip(self.mock_rule_types, rule_types):
        self._compare_rule_types(a, b)
    self.assert_calls()