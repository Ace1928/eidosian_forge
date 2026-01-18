from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_creation_with_version(self):
    version = (2, 0)
    f_task = task.FunctorTask(lambda: None, version=version)
    self.assertEqual(version, f_task.version)