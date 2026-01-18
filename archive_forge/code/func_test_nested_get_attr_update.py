from heat_integrationtests.functional import functional_base
def test_nested_get_attr_update(self):
    stack_identifier = self.stack_create(template=initial_template)
    self.update_stack(stack_identifier, template=update_template)
    self.assertOutput('bar', stack_identifier, 'value1')
    self.assertOutput('barney', stack_identifier, 'value2')
    self.assertOutput('wibble', stack_identifier, 'value3')
    self.assertOutput('quux', stack_identifier, 'value4')