from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_multi_provides(self):
    my_task = MyTask(provides=('food', 'water'))
    self.assertEqual({'food': 0, 'water': 1}, my_task.save_as)