from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
def test_include_stack_status_with_no_action(self):
    stack = {'stack_status': 'COMPLETE'}
    result = stacks_view.format_stack(self.request, stack)
    self.assertIn('stack_status', result)
    self.assertEqual('COMPLETE', result['stack_status'])