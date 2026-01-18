import contextlib
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_memory
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_del_no_children_allowed(self):
    fs = impl_memory.FakeFilesystem()
    fs['/a'] = 'a'
    self.assertEqual(1, len(fs.ls_r('/')))
    fs.delete('/a')
    self.assertEqual(0, len(fs.ls('/')))