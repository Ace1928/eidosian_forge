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
def test_call_no_working_seek(self):
    """Similar to 'test_call_no_seek', but this time the methods are available
        (but defunct).  See https://github.com/jonashaag/klaus/issues/154.
        """
    zstream, zlength = self._get_zstream(self.example_text)
    self._test_call(self.example_text, MinimalistWSGIInputStream2(zstream.read()), zlength)