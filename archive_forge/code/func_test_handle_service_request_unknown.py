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
def test_handle_service_request_unknown(self):
    mat = re.search('.*', '/git-evil-handler')
    content = list(handle_service_request(self._req, 'backend', mat))
    self.assertEqual(HTTP_FORBIDDEN, self._status)
    self.assertNotIn(b'git-evil-handler', b''.join(content))
    self.assertFalse(self._req.cached)