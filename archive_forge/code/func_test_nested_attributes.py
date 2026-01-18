import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
def test_nested_attributes(self):
    nested_templ = '\nheat_template_version: 2014-10-16\nresources:\n  secret1:\n    type: OS::Heat::RandomString\noutputs:\n  nested_str:\n    value: {get_attr: [secret1, value]}\n'
    stack_identifier = self.stack_create(template=self.main_templ, files={'nested.yaml': nested_templ}, environment=self.env_templ)
    self.assert_resource_is_a_stack(stack_identifier, 'secret2')
    stack = self.client.stacks.get(stack_identifier)
    old_way = self._stack_output(stack, 'old_way')
    test_attr1 = self._stack_output(stack, 'test_attr1')
    test_attr2 = self._stack_output(stack, 'test_attr2')
    self.assertEqual(old_way, test_attr1)
    self.assertEqual(old_way, test_attr2)