import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_inplace_update_old_ref_deleted_failed_stack(self):
    template = '\nheat_template_version: rocky\nresources:\n  test1:\n    type: OS::Heat::TestResource\n    properties:\n      value: test\n  test2:\n    type: OS::Heat::TestResource\n    properties:\n      value: {get_attr: [test1, output]}\n  test3:\n    type: OS::Heat::TestResource\n    properties:\n      value: test3\n      fail: false\n      action_wait_secs:\n        update: 5\n'
    stack_identifier = self.stack_create(template=template)
    _template = template.replace('test1:', 'test-1:').replace('fail: false', 'fail: true')
    updated_template = _template.replace('{get_attr: [test1', '{get_attr: [test-1').replace('value: test3', 'value: test-3')
    self.update_stack(stack_identifier, template=updated_template, expected_status='UPDATE_FAILED')
    self.update_stack(stack_identifier, template=template, expected_status='UPDATE_COMPLETE')