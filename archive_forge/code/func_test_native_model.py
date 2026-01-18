from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
def test_native_model(self):
    model_cls = glance.async_.NativeThreadPoolModel
    self.assertEqual(futurist.ThreadPoolExecutor, model_cls.get_threadpool_executor_class())