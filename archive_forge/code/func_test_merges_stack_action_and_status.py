from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
def test_merges_stack_action_and_status(self):
    stack = {'stack_action': 'CREATE', 'stack_status': 'COMPLETE'}
    result = stacks_view.format_stack(self.request, stack)
    self.assertIn('stack_status', result)
    self.assertEqual('CREATE_COMPLETE', result['stack_status'])