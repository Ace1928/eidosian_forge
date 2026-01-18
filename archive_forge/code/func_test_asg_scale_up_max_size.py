from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_asg_scale_up_max_size(self):
    stack_id = self.stack_create(template=self.template, expected_status='CREATE_COMPLETE')
    stack = self.client.stacks.get(stack_id)
    asg_size = self._stack_output(stack, 'asg_size')
    self.assertEqual(3, asg_size)
    asg = self.client.resources.get(stack_id, 'random_group')
    max_size = 5
    for num in range(asg_size + 1, max_size + 2):
        expected_resources = num if num <= max_size else max_size
        self.client.resources.signal(stack_id, 'scale_up_policy')
        self.assertTrue(test.call_until_true(self.conf.build_timeout, self.conf.build_interval, self.check_autoscale_complete, asg.physical_resource_id, expected_resources, stack_id, 'random_group'))