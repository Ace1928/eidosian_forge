from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_no_left_ref(self):
    r = {b'refs/heads/foo': 'bla'}
    self.assertEqual((None, b'refs/heads/foo', False), parse_reftuple(r, r, b':refs/heads/foo'))