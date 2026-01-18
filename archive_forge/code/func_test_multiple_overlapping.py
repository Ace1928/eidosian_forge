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
def test_multiple_overlapping(self):
    c1, c2 = self.make_linear_commits(2)
    c3 = self.make_commit(parents=[c1.id])
    c4 = self.make_commit(parents=[c3.id])
    self.assertEqual(({c1.id}, {c1.id, c2.id, c3.id, c4.id}), _find_shallow(self._store, [c2.id, c4.id], 3))