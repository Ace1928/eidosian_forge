import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
def test_new_task_executor_with_admin(self):
    admin_repo = mock.MagicMock()
    task_executor_factory = domain.TaskExecutorFactory(self.task_repo, self.image_repo, self.image_factory, admin_repo=admin_repo)
    context = mock.Mock()
    with mock.patch.object(oslo_utils.importutils, 'import_class') as mock_import_class:
        mock_executor = mock.Mock()
        mock_import_class.return_value = mock_executor
        task_executor_factory.new_task_executor(context)
    mock_executor.assert_called_once_with(context, self.task_repo, self.image_repo, self.image_factory, admin_repo=admin_repo)