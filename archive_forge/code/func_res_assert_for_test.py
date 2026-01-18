from heat_integrationtests.functional import functional_base
def res_assert_for_test(self, stack_identifier, fj_zone=False, shannxi_provice=False):

    def is_not_deleted(r):
        return r.resource_status != 'DELETE_COMPLETE'
    resources = self.list_resources(stack_identifier, is_not_deleted)
    res_names = set(resources)
    if fj_zone:
        self.assertEqual(4, len(resources))
        self.assertIn('fujian_res', res_names)
        self.assertIn('not_shannxi_res', res_names)
    elif shannxi_provice:
        self.assertEqual(3, len(resources))
        self.assertNotIn('fujian_res', res_names)
        self.assertIn('shannxi_res', res_names)
    else:
        self.assertEqual(3, len(resources))
        self.assertIn('not_shannxi_res', res_names)
    self.assertIn('test_res', res_names)
    self.assertIn('test_res1', res_names)
    self.assertNotIn('prod_res', res_names)