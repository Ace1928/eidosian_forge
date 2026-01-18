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
def test_get_loose_object_missing(self):
    mat = re.search('^(..)(.{38})$', '1' * 40)
    list(get_loose_object(self._req, _test_backend([]), mat))
    self.assertEqual(HTTP_NOT_FOUND, self._status)