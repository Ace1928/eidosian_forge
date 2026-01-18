from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_ambiguous_remote(self):
    r = {b'refs/remotes/ambig5': 'bla', b'refs/remotes/ambig5/HEAD': 'bla'}
    self.assertEqual(b'refs/remotes/ambig5', parse_ref(r, b'ambig5'))