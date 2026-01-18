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
def test_cached_retry_failure(self):
    a_failure = failure.Failure.from_exception(RuntimeError('Woot!'))
    s = self._get_storage()
    s.ensure_atom(test_utils.NoopRetry('my retry', provides=['x']))
    s.save('my retry', 'a')
    s.save('my retry', a_failure, states.FAILURE)
    history = s.get_retry_history('my retry')
    self.assertEqual([('a', {})], list(history))
    self.assertTrue(history.caused_by(RuntimeError, include_retry=True))
    self.assertIsNotNone(history.failure)
    self.assertEqual(1, len(history))
    self.assertTrue(s.has_failures())
    self.assertEqual({'my retry': a_failure}, s.get_failures())