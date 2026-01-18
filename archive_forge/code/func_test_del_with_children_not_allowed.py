import contextlib
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_memory
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_del_with_children_not_allowed(self):
    fs = impl_memory.FakeFilesystem()
    fs['/a'] = 'a'
    fs['/a/b'] = 'b'
    self.assertRaises(ValueError, fs.delete, '/a', recursive=False)