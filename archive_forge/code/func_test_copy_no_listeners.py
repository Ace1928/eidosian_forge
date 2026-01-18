from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_copy_no_listeners(self):
    handler1 = lambda event_type, details: None
    a_task = MyTask()
    a_task.notifier.register(task.EVENT_UPDATE_PROGRESS, handler1)
    b_task = a_task.copy(retain_listeners=False)
    self.assertEqual(1, len(a_task.notifier))
    self.assertEqual(0, len(b_task.notifier))