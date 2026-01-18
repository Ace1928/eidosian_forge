from heat_integrationtests.functional import functional_base
def test_nested_resources_add(self):
    files = {'nested1.yaml': self.template_nested1, 'nested2.yaml': self.template_nested2}
    self.stack_identifier = self.stack_create(template=self.template_nested_parent, files=files)
    files['nested2.yaml'] = self.template_nested2_2
    result = self.preview_update_stack(self.stack_identifier, template=self.template_nested_parent, files=files, show_nested=True)
    changes = result['resource_changes']
    self.assertEqual(1, len(changes['unchanged']))
    self.assertEqual('random', changes['unchanged'][0]['resource_name'])
    self.assertEqual('nested2', changes['unchanged'][0]['parent_resource'])
    self.assertEqual(1, len(changes['added']))
    self.assertEqual('random2', changes['added'][0]['resource_name'])
    self.assertEqual('nested2', changes['added'][0]['parent_resource'])
    self.assert_empty_sections(changes, ['replaced', 'deleted'])