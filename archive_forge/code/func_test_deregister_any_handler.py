from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_deregister_any_handler(self):
    a_task = MyTask()
    self.assertEqual(0, len(a_task.notifier))
    a_task.notifier.register(task.EVENT_UPDATE_PROGRESS, lambda event_type, details: None)
    self.assertEqual(1, len(a_task.notifier))
    a_task.notifier.deregister_event(task.EVENT_UPDATE_PROGRESS)
    self.assertEqual(0, len(a_task.notifier))