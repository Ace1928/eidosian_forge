from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
@mock.patch('glance.async_.ThreadPoolModel.get_threadpool_executor_class')
def test_base_model_spawn(self, mock_gte):
    pool_cls = mock.MagicMock()
    pool_cls.configure_mock(__name__='fake')
    mock_gte.return_value = pool_cls
    model = glance.async_.ThreadPoolModel()
    result = model.spawn(print, 'foo', bar='baz')
    pool = pool_cls.return_value
    pool_cls.assert_called_once_with(1)
    pool.submit.assert_called_once_with(print, 'foo', bar='baz')
    self.assertEqual(pool.submit.return_value, result)