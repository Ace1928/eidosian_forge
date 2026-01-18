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
def test_get_info_packs(self):

    class TestPackData:

        def __init__(self, sha) -> None:
            self.filename = 'pack-%s.pack' % sha

    class TestPack:

        def __init__(self, sha) -> None:
            self.data = TestPackData(sha)
    packs = [TestPack(str(i) * 40) for i in range(1, 4)]

    class TestObjectStore(MemoryObjectStore):

        @property
        def packs(self):
            return packs
    store = TestObjectStore()
    repo = BaseRepo(store, None)
    backend = DictBackend({'/': repo})
    mat = re.search('.*', '//info/packs')
    output = b''.join(get_info_packs(self._req, backend, mat))
    expected = b''.join([b'P pack-' + s + b'.pack\n' for s in [b'1' * 40, b'2' * 40, b'3' * 40]])
    self.assertEqual(expected, output)
    self.assertEqual(HTTP_OK, self._status)
    self.assertContentTypeEquals('text/plain')
    self.assertFalse(self._req.cached)