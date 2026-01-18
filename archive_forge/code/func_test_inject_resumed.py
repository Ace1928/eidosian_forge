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
def test_inject_resumed(self):
    s = self._get_storage()
    s.inject({'foo': 'bar', 'spam': 'eggs'})
    self.assertEqual({'foo': 'bar', 'spam': 'eggs'}, s.fetch_all())
    s2 = self._get_storage(s._flowdetail)
    self.assertEqual({'foo': 'bar', 'spam': 'eggs'}, s2.fetch_all())