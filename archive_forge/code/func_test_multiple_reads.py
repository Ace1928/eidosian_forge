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
def test_multiple_reads(self):
    f = _LengthLimitedFile(BytesIO(b'foobar'), 3)
    self.assertEqual(b'fo', f.read(2))
    self.assertEqual(b'o', f.read(2))
    self.assertEqual(b'', f.read())