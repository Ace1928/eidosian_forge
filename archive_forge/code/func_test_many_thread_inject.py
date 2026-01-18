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
def test_many_thread_inject(self):
    s = self._get_storage()

    def inject_values(values):
        s.inject(values)
    threads = []
    for i in range(0, self.thread_count):
        values = {str(i): str(i)}
        threads.append(threading.Thread(target=inject_values, args=[values]))
    self._run_many_threads(threads)
    self.assertEqual(self.thread_count, len(s.fetch_all()))
    self.assertEqual(1, len(s._flowdetail))