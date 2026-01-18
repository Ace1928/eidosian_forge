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
def test_get_pack_file(self):
    pack_name = os.path.join('objects', 'pack', 'pack-%s.pack' % ('1' * 40))
    backend = _test_backend([], named_files={pack_name: b'pack contents'})
    mat = re.search('.*', pack_name)
    output = b''.join(get_pack_file(self._req, backend, mat))
    self.assertEqual(b'pack contents', output)
    self.assertEqual(HTTP_OK, self._status)
    self.assertContentTypeEquals('application/x-git-packed-objects')
    self.assertTrue(self._req.cached)