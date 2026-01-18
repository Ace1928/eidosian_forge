import contextlib
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_memory
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_ls_recursive_absolute(self):
    fs = impl_memory.FakeFilesystem()
    fs.ensure_path('/d')
    fs.ensure_path('/c/d')
    fs.ensure_path('/b/c/d')
    fs.ensure_path('/a/b/c/d')
    contents = fs.ls_r('/', absolute=True)
    self.assertEqual(['/a', '/b', '/c', '/d', '/a/b', '/b/c', '/c/d', '/a/b/c', '/b/c/d', '/a/b/c/d'], contents)