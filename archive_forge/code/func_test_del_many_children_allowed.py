import contextlib
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_memory
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_del_many_children_allowed(self):
    fs = impl_memory.FakeFilesystem()
    fs['/a'] = 'a'
    fs['/a/b'] = 'b'
    self.assertEqual(2, len(fs.ls_r('/')))
    fs.delete('/a', recursive=True)
    self.assertEqual(0, len(fs.ls('/')))