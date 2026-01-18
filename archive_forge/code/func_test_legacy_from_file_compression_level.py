import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def test_legacy_from_file_compression_level(self):
    b1 = Blob.from_string(b'foo')
    b_raw = b1.as_legacy_object(compression_level=6)
    b2 = b1.from_file(BytesIO(b_raw))
    self.assertEqual(b1, b2)