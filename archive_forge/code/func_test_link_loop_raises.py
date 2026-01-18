import contextlib
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_memory
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_link_loop_raises(self):
    fs = impl_memory.FakeFilesystem()
    fs['/b'] = 'c'
    fs.symlink('/b', '/b')
    self.assertRaises(ValueError, self._get_item_path, fs, '/b')