from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_functor_invalid_requires(self):
    self.assertRaises(TypeError, task.MapFunctorTask, lambda x: None, requires=1)