from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
def test_model_map(self):
    model = glance.async_.EventletThreadPoolModel()
    results = model.map(lambda s: s.upper(), ['a', 'b', 'c'])
    self.assertEqual(['A', 'B', 'C'], list(results))