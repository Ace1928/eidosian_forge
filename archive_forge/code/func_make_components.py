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
def make_components(self):
    client = fake_client.FakeClient()
    persistence = impl_memory.MemoryBackend()
    board = impl_zookeeper.ZookeeperJobBoard('testing', {}, client=client, persistence=persistence)
    conductor_kwargs = self.conductor_kwargs.copy()
    conductor_kwargs['persistence'] = persistence
    conductor = backends.fetch(self.kind, 'testing', board, **conductor_kwargs)
    return ComponentBundle(board, client, persistence, conductor)