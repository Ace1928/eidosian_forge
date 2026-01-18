import collections
import contextlib
import threading
import futurist
import testscenarios
from zake import fake_client
from taskflow.conductors import backends
from taskflow import engines
from taskflow.jobs.backends import impl_zookeeper
from taskflow.jobs import base
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as pu
from taskflow.utils import threading_utils
def test_bad_factory(self):
    persistence = impl_memory.MemoryBackend()
    client = fake_client.FakeClient()
    board = impl_zookeeper.ZookeeperJobBoard('testing', {}, client=client, persistence=persistence)
    self.assertRaises(ValueError, backends.fetch, 'nonblocking', 'testing', board, persistence=persistence, executor_factory='testing')