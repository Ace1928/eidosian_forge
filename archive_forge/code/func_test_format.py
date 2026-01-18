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
def test_format(self):
    self.assertEqual('40000 tree 40820c38cfb182ce6c8b261555410d8382a5918b\tfoo\n', pretty_format_tree_entry(b'foo', 16384, b'40820c38cfb182ce6c8b261555410d8382a5918b'))