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
def test_custom(self):
    c = Commit.from_string(self.make_commit_text(extra={b'extra-field': b'data'}))
    self.assertEqual([(b'extra-field', b'data')], c._extra)