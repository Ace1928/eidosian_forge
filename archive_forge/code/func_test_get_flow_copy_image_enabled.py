from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
@mock.patch.object(store, 'get_store_from_store_identifier')
def test_get_flow_copy_image_enabled(self, mock_store):
    import_req = {'method': {'name': 'copy-image', 'stores': ['fake-store']}}
    mock_store.return_value = mock.Mock()
    flow = self._get_flow(import_req=import_req)
    flow_comp = self._get_flow_tasks(flow)
    self.assertEqual(len(self.base_flow) + 1, len(flow_comp))
    for c in self.base_flow:
        self.assertIn(c, flow_comp)
    self.assertIn('CopyImage', flow_comp)