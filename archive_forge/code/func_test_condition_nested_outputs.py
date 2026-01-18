from heat_integrationtests.functional import functional_base
def test_condition_nested_outputs(self):
    stack_identifier = self.stack_create(template=root_output_tmpl, files={'nested_output.yaml': nested_output_tmpl})
    standard = self.client.stacks.output_show(stack_identifier, 'standard')['output']
    self.assertEqual('hello', standard['output_value'])
    cond = self.client.stacks.output_show(stack_identifier, 'cond')['output']
    self.assertIsNone(cond['output_value'])
    cond_val = self.client.stacks.output_show(stack_identifier, 'cond_value')['output']
    self.assertEqual('test', cond_val['output_value'])