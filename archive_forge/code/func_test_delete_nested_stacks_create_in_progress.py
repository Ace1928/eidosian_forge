import time
from heat_integrationtests.functional import functional_base
def test_delete_nested_stacks_create_in_progress(self):
    files = {'empty.yaml': self.empty_template}
    identifier = self.stack_create(template=self.root_template, files=files, expected_status='CREATE_IN_PROGRESS')
    time.sleep(20)
    self._stack_delete(identifier)