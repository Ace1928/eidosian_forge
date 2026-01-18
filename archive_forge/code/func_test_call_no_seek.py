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
def test_call_no_seek(self):
    """This ensures that the gunzipping code doesn't require any methods on
        'wsgi.input' except for '.read()'.  (In particular, it shouldn't
        require '.seek()'. See https://github.com/jelmer/dulwich/issues/140.).
        """
    zstream, zlength = self._get_zstream(self.example_text)
    self._test_call(self.example_text, MinimalistWSGIInputStream(zstream.read()), zlength)