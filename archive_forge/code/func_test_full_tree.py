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
def test_full_tree(self):
    c = self.make_commit(commit_time=30)
    t = Tree()
    t.add(b'data-x', 420, Blob().id)
    c.tree = t
    c1 = Commit()
    c1.set_raw_string(c.as_raw_string())
    self.assertEqual(t.id, c1.tree)
    self.assertEqual(c.as_raw_string(), c1.as_raw_string())