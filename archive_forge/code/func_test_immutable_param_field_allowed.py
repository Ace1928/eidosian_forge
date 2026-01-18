from heat_integrationtests.functional import functional_base
from heatclient import exc as heat_exceptions
def test_immutable_param_field_allowed(self):
    param1_create_value = 'value1'
    create_parameters = {'param1': param1_create_value}
    stack_identifier = self.stack_create(template=self.template_param_has_immutable_field, parameters=create_parameters)
    stack = self.client.stacks.get(stack_identifier)
    self.assertEqual(param1_create_value, self._stack_output(stack, 'param1_output'))
    param1_update_value = 'value2'
    update_parameters = {'param1': param1_update_value}
    self.update_stack(stack_identifier, template=self.template_param_has_immutable_field, parameters=update_parameters)
    stack = self.client.stacks.get(stack_identifier)
    self.assertEqual(param1_update_value, self._stack_output(stack, 'param1_output'))
    self.assertEqual('UPDATE_COMPLETE', stack.stack_status)