from heat_integrationtests.functional import functional_base
def test_delete_resource(self):
    self.stack_identifier = self.stack_create(template=test_template_two_resource)
    result = self.preview_update_stack(self.stack_identifier, test_template_one_resource)
    changes = result['resource_changes']
    unchanged = changes['unchanged'][0]['resource_name']
    self.assertEqual('test1', unchanged)
    deleted = changes['deleted'][0]['resource_name']
    self.assertEqual('test2', deleted)
    self.assert_empty_sections(changes, ['updated', 'replaced', 'added'])