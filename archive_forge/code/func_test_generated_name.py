from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_generated_name(self):
    my_task = MyTask()
    self.assertEqual('%s.%s' % (__name__, 'MyTask'), my_task.name)