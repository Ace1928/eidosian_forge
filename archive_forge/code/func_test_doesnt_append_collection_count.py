from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
@mock.patch.object(stacks_view.views_common, 'get_collection_links')
def test_doesnt_append_collection_count(self, mock_get_collection_links):
    stacks = [self.stack1]
    stack_view = stacks_view.collection(self.request, stacks)
    self.assertNotIn('count', stack_view)