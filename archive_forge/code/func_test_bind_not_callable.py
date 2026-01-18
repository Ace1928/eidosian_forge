from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_bind_not_callable(self):
    a_task = MyTask()
    self.assertRaises(ValueError, a_task.notifier.register, task.EVENT_UPDATE_PROGRESS, 2)