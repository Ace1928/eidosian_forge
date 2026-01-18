from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
def test_get_flow_with_all_plugins_enabled(self):
    self.config(image_import_plugins=['image_conversion', 'image_decompression', 'inject_image_metadata'], group='image_import_opts')
    flow = self._get_flow()
    flow_comp = self._get_flow_tasks(flow)
    plugins = CONF.image_import_opts.image_import_plugins
    self.assertEqual(len(self.base_flow) + len(plugins), len(flow_comp))
    for c in self.base_flow:
        self.assertIn(c, flow_comp)
    for c in self.import_plugins:
        self.assertIn(c, flow_comp)