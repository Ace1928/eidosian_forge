from heat_integrationtests.functional import functional_base
def test_create_update_translation_with_get_attr(self):
    env = {'parameters': {'net_cidr': '11.11.11.0/24'}}
    stack_identifier = self.stack_create(template=template_with_get_attr, environment=env)
    env = {'parameters': {'net_cidr': '11.11.12.0/24'}}
    self.update_stack(stack_identifier, template=template_with_get_attr, environment=env)