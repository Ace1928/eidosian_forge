from unittest import mock
import futurist
import glance_store
from oslo_config import cfg
from taskflow import engines
import glance.async_
from glance.async_ import taskflow_executor
from glance.common.scripts.image_import import main as image_import
from glance import domain
import glance.tests.utils as test_utils
@mock.patch('stevedore.driver.DriverManager')
@mock.patch.object(taskflow_executor, 'LOG')
def test_get_flow_fails(self, mock_log, mock_driver):
    mock_driver.side_effect = IndexError('fail')
    executor = taskflow_executor.TaskExecutor(self.context, self.task_repo, self.image_repo, self.image_factory)
    self.assertRaises(IndexError, executor._get_flow, self.task)
    mock_log.exception.assert_called_once_with('Task initialization failed: %s', 'fail')