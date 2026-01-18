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
def test_get_info_refs_unknown(self):
    self._environ['QUERY_STRING'] = 'service=git-evil-handler'

    class Backend:

        def open_repository(self, url):
            return None
    mat = re.search('.*', '/git-evil-pack')
    content = list(get_info_refs(self._req, Backend(), mat))
    self.assertNotIn(b'git-evil-handler', b''.join(content))
    self.assertEqual(HTTP_FORBIDDEN, self._status)
    self.assertFalse(self._req.cached)