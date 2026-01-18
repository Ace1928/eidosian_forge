import contextlib
import threading
from oslo_utils import uuidutils
from taskflow import exceptions
from taskflow.persistence import backends
from taskflow.persistence import models
from taskflow import states
from taskflow import storage
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
from taskflow.utils import persistence_utils as p_utils
def test_get_tasks_states(self):
    s = self._get_storage()
    s.ensure_atom(test_utils.NoopTask('my task'))
    s.ensure_atom(test_utils.NoopTask('my task2'))
    s.save('my task', 'foo')
    expected = {'my task': (states.SUCCESS, states.EXECUTE), 'my task2': (states.PENDING, states.EXECUTE)}
    self.assertEqual(expected, s.get_atoms_states(['my task', 'my task2']))