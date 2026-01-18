from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_extract_spaces(self):
    text = b'From ff643aae102d8870cac88e8f007e70f58f3a7363 Mon Sep 17 00:00:00 2001\nFrom: Jelmer Vernooij <jelmer@samba.org>\nDate: Thu, 15 Apr 2010 15:40:28 +0200\nSubject:  [Dulwich-users] [PATCH] Added unit tests for\n dulwich.object_store.tree_lookup_path.\n\n* dulwich/tests/test_object_store.py\n  (TreeLookupPathTests): This test case contains a few tests that ensure the\n   tree_lookup_path function works as expected.\n---\n pixmaps/prey.ico |  Bin 9662 -> 9662 bytes\n 1 files changed, 0 insertions(+), 0 deletions(-)\n mode change 100755 => 100644 pixmaps/prey.ico\n\n-- \n1.7.0.4\n'
    c, diff, version = git_am_patch_split(BytesIO(text), 'utf-8')
    self.assertEqual(b'Added unit tests for dulwich.object_store.tree_lookup_path.\n\n* dulwich/tests/test_object_store.py\n  (TreeLookupPathTests): This test case contains a few tests that ensure the\n   tree_lookup_path function works as expected.\n', c.message)