from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
@mock.patch.object(stacks_view.views_common, 'get_collection_links')
def test_append_collection_count(self, mock_get_collection_links):
    stacks = [self.stack1]
    count = 1
    stack_view = stacks_view.collection(self.request, stacks, count)
    self.assertIn('count', stack_view)
    self.assertEqual(1, stack_view['count'])