from heat_integrationtests.functional import functional_base
def output_assert_for_test(self, stack_id):
    output = self.client.stacks.output_show(stack_id, 'res_value')['output']
    self.assertIsNone(output['output_value'])
    test_res_value = self.client.stacks.output_show(stack_id, 'test_res_value')['output']
    self.assertEqual('env_is_test', test_res_value['output_value'])
    prod_resource = self.client.stacks.output_show(stack_id, 'prod_resource')['output']
    self.assertEqual('no_prod_res', prod_resource['output_value'])
    test_res_output = self.client.stacks.output_show(stack_id, 'test_res1_value')['output']
    self.assertEqual('just in test env', test_res_output['output_value'])
    beijing_prod_res = self.client.stacks.output_show(stack_id, 'beijing_prod_res')['output']
    self.assertEqual('no_prod_res', beijing_prod_res['output_value'])