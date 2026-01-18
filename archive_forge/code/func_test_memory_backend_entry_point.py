import contextlib
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_memory
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_memory_backend_entry_point(self):
    conf = {'connection': 'memory:'}
    with contextlib.closing(backends.fetch(conf)) as be:
        self.assertIsInstance(be, impl_memory.MemoryBackend)