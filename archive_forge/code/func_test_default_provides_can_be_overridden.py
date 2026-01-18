from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_default_provides_can_be_overridden(self):
    my_task = DefaultProvidesTask(provides=('spam', 'eggs'))
    self.assertEqual(set(['spam', 'eggs']), my_task.provides)
    self.assertEqual({'spam': 0, 'eggs': 1}, my_task.save_as)