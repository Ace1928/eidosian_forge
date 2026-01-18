from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_full(self):
    r = {b'refs/heads/foo': 'bla'}
    self.assertEqual([(b'refs/heads/foo', b'refs/heads/foo', False)], parse_reftuples(r, r, b'refs/heads/foo'))
    r = {b'refs/heads/foo': 'bla'}
    self.assertEqual([(b'refs/heads/foo', b'refs/heads/foo', True)], parse_reftuples(r, r, b'refs/heads/foo', True))