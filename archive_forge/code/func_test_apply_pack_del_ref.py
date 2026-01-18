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
def test_apply_pack_del_ref(self):
    refs = {b'refs/heads/master': TWO, b'refs/heads/fake-branch': ONE}
    self._repo.refs._update(refs)
    update_refs = [[ONE, ZERO_SHA, b'refs/heads/fake-branch']]
    self._handler.set_client_capabilities([b'delete-refs'])
    status = self._handler._apply_pack(update_refs)
    self.assertEqual(status[0][0], b'unpack')
    self.assertEqual(status[0][1], b'ok')
    self.assertEqual(status[1][0], b'refs/heads/fake-branch')
    self.assertEqual(status[1][1], b'ok')