from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_ambiguous_remote_head(self):
    r = {b'refs/remotes/ambig6/HEAD': 'bla'}
    self.assertEqual(b'refs/remotes/ambig6/HEAD', parse_ref(r, b'ambig6'))