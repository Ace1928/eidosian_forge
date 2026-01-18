import os
import shutil
import sys
import tempfile
from io import BytesIO
from typing import Dict, List
from dulwich.tests import TestCase
from ..errors import (
from ..object_store import MemoryObjectStore
from ..objects import Tree
from ..protocol import ZERO_SHA, format_capability_line
from ..repo import MemoryRepo, Repo
from ..server import (
from .utils import make_commit, make_tag
def test_multiple_independent(self):
    a = self.make_linear_commits(2, message=b'a')
    b = self.make_linear_commits(2, message=b'b')
    c = self.make_linear_commits(2, message=b'c')
    heads = [a[1].id, b[1].id, c[1].id]
    self.assertEqual(({a[0].id, b[0].id, c[0].id}, set(heads)), _find_shallow(self._store, heads, 2))