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
def test_get_loose_object_error(self):
    blob = make_object(Blob, data=b'foo')
    backend = _test_backend([blob])
    mat = re.search('^(..)(.{38})$', blob.id.decode('ascii'))

    def as_legacy_object_error(self):
        raise OSError
    self.addCleanup(setattr, Blob, 'as_legacy_object', Blob.as_legacy_object)
    Blob.as_legacy_object = as_legacy_object_error
    list(get_loose_object(self._req, backend, mat))
    self.assertEqual(HTTP_ERROR, self._status)