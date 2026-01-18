from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_execute_not_callable(self):
    self.assertRaises(ValueError, task.FunctorTask, 2)