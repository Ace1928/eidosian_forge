from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
def test_get_flow_web_download_enabled(self):
    import_req = {'method': {'name': 'web-download', 'uri': 'http://cloud.foo/image.qcow2'}}
    flow = self._get_flow(import_req=import_req)
    flow_comp = self._get_flow_tasks(flow)
    self.assertEqual(len(self.base_flow) + 1, len(flow_comp))
    for c in self.base_flow:
        self.assertIn(c, flow_comp)
    self.assertIn('WebDownload', flow_comp)