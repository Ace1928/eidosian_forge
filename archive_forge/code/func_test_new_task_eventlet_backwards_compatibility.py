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
def test_new_task_eventlet_backwards_compatibility(self):
    context = mock.MagicMock()
    self.config(task_executor='eventlet', group='task')
    task_executor_factory = domain.TaskExecutorFactory(self.task_repo, self.image_repo, self.image_factory)
    te_evnt = task_executor_factory.new_task_executor(context)
    self.assertIsInstance(te_evnt, taskflow_executor.TaskExecutor)