from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_no_provides(self):
    my_task = MyTask()
    self.assertEqual({}, my_task.save_as)