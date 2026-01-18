import contextlib
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_memory
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_bad_norms(self):
    fs = impl_memory.FakeFilesystem()
    self.assertRaises(ValueError, fs.normpath, '')
    self.assertRaises(ValueError, fs.normpath, 'abc/c')
    self.assertRaises(ValueError, fs.normpath, '../c')