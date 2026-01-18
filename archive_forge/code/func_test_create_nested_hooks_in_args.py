from unittest import mock
import testtools
from heatclient.common import hook_utils
import heatclient.v1.shell as shell
def test_create_nested_hooks_in_args(self):
    type(self.args).pre_create = mock.PropertyMock(return_value=['nested/bp', 'super/nested/bp'])
    shell.do_stack_create(self.client, self.args)
    self.assertEqual(1, self.client.stacks.create.call_count)
    expected_hooks = {'nested': {'bp': {'hooks': 'pre-create'}}, 'super': {'nested': {'bp': {'hooks': 'pre-create'}}}}
    actual_hooks = self.client.stacks.create.call_args[1]['environment']['resource_registry']['resources']
    self.assertEqual(expected_hooks, actual_hooks)