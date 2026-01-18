from heat_integrationtests.functional import functional_base
def test_nested_get_attr_create(self):
    stack_identifier = self.stack_create(template=initial_template)
    self.assertOutput('wibble', stack_identifier, 'value1')
    self.assertOutput('bar', stack_identifier, 'value2')
    self.assertOutput('barney', stack_identifier, 'value3')