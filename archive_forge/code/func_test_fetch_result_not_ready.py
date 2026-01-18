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
def test_fetch_result_not_ready(self):
    s = self._get_storage()
    name = 'my result'
    s.ensure_atom(test_utils.NoopTask('my task', provides=name))
    self.assertRaises(exceptions.NotFound, s.get, name)
    self.assertEqual({}, s.fetch_all())