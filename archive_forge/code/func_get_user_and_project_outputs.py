from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def get_user_and_project_outputs(self, stack_identifier):
    stack = self.client.stacks.get(stack_identifier)
    project_name = self._stack_output(stack, 'project_name')
    user_name = self._stack_output(stack, 'user_name')
    return (project_name, user_name)