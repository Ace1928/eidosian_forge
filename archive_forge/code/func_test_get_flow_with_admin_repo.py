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
def test_get_flow_with_admin_repo(self, mock_driver):
    admin_repo = mock.MagicMock()
    executor = taskflow_executor.TaskExecutor(self.context, self.task_repo, self.image_repo, self.image_factory, admin_repo=admin_repo)
    self.assertEqual(mock_driver.return_value.driver, executor._get_flow(self.task))
    mock_driver.assert_called_once_with('glance.flows', self.task.type, invoke_on_load=True, invoke_kwds={'task_id': self.task.task_id, 'task_type': self.task.type, 'context': self.context, 'task_repo': self.task_repo, 'image_repo': self.image_repo, 'image_factory': self.image_factory, 'backend': None, 'admin_repo': admin_repo, 'uri': 'http://cloud.foo/image.qcow2'})