from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_asg_cooldown(self):
    cooldown_tmpl = self.template.replace('cooldown: 0', 'cooldown: 60')
    stack_id = self.stack_create(template=cooldown_tmpl, expected_status='CREATE_COMPLETE')
    stack = self.client.stacks.get(stack_id)
    asg_size = self._stack_output(stack, 'asg_size')
    self.assertEqual(3, asg_size)
    asg = self.client.resources.get(stack_id, 'random_group')
    expected_resources = 3
    self.client.resources.signal(stack_id, 'scale_up_policy')
    self.assertTrue(test.call_until_true(self.conf.build_timeout, self.conf.build_interval, self.check_autoscale_complete, asg.physical_resource_id, expected_resources, stack_id, 'random_group'))