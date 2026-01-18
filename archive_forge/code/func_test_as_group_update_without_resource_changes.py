from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_as_group_update_without_resource_changes(self):
    stack_identifier = self.stack_create(template=self.template)
    new_template = self.template.replace('scaling_adjustment: 1', 'scaling_adjustment: 2')
    self.update_stack(stack_identifier, template=new_template)