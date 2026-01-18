import gzip
import os
import re
from io import BytesIO
from typing import Type
from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore
from ..objects import Blob
from ..repo import BaseRepo, MemoryRepo
from ..server import DictBackend
from ..web import (
from .utils import make_object, make_tag
def test_get_idx_file(self):
    idx_name = os.path.join('objects', 'pack', 'pack-%s.idx' % ('1' * 40))
    backend = _test_backend([], named_files={idx_name: b'idx contents'})
    mat = re.search('.*', idx_name)
    output = b''.join(get_idx_file(self._req, backend, mat))
    self.assertEqual(b'idx contents', output)
    self.assertEqual(HTTP_OK, self._status)
    self.assertContentTypeEquals('application/x-git-packed-objects-toc')
    self.assertTrue(self._req.cached)