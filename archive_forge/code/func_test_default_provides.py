from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_default_provides(self):
    my_task = DefaultProvidesTask()
    self.assertEqual(set(['def']), my_task.provides)
    self.assertEqual({'def': None}, my_task.save_as)