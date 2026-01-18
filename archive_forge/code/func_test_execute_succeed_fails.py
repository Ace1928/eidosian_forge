import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
def test_execute_succeed_fails(self, mock_get_task):
    mock_get_task.return_value = self.task
    self.task.succeed.side_effect = Exception('testing')
    complete = import_flow._CompleteTask(TASK_ID1, TASK_TYPE, self.task_repo, self.wrapper)
    complete.execute()
    self.task.fail.assert_called_once_with(_("Error: <class 'Exception'>: testing"))
    self.task_repo.save.assert_called_once_with(self.task)
    self.wrapper.drop_lock_for_task.assert_called_once_with()