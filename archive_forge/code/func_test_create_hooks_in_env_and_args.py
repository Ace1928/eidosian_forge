from unittest import mock
import testtools
from heatclient.common import hook_utils
import heatclient.v1.shell as shell
def test_create_hooks_in_env_and_args(self):
    type(self.args).pre_create = mock.PropertyMock(return_value=['nested_a/bp', 'bp_a', 'another_bp_a', 'super_a/nested/bp'])
    env = {'resource_registry': {'resources': {'bp_e': {'hooks': 'pre-create'}, 'another_bp_e': {'hooks': 'pre-create'}, 'nested_e': {'bp': {'hooks': 'pre-create'}}, 'super_e': {'nested': {'bp': {'hooks': 'pre-create'}}}}}}
    shell.template_utils.process_multiple_environments_and_files = mock.Mock(return_value=({}, env))
    shell.do_stack_create(self.client, self.args)
    self.assertEqual(1, self.client.stacks.create.call_count)
    actual_hooks = self.client.stacks.create.call_args[1]['environment']['resource_registry']['resources']
    expected_hooks = {'bp_e': {'hooks': 'pre-create'}, 'another_bp_e': {'hooks': 'pre-create'}, 'nested_e': {'bp': {'hooks': 'pre-create'}}, 'super_e': {'nested': {'bp': {'hooks': 'pre-create'}}}, 'bp_a': {'hooks': 'pre-create'}, 'another_bp_a': {'hooks': 'pre-create'}, 'nested_a': {'bp': {'hooks': 'pre-create'}}, 'super_a': {'nested': {'bp': {'hooks': 'pre-create'}}}}
    self.assertEqual(expected_hooks, actual_hooks)