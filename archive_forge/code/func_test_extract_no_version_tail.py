from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_extract_no_version_tail(self):
    text = b'From ff643aae102d8870cac88e8f007e70f58f3a7363 Mon Sep 17 00:00:00 2001\nFrom: Jelmer Vernooij <jelmer@samba.org>\nDate: Thu, 15 Apr 2010 15:40:28 +0200\nSubject:  [Dulwich-users] [PATCH] Added unit tests for\n dulwich.object_store.tree_lookup_path.\n\nFrom: Jelmer Vernooij <jelmer@debian.org>\n\n---\n pixmaps/prey.ico |  Bin 9662 -> 9662 bytes\n 1 files changed, 0 insertions(+), 0 deletions(-)\n mode change 100755 => 100644 pixmaps/prey.ico\n\n'
    c, diff, version = git_am_patch_split(BytesIO(text), 'utf-8')
    self.assertEqual(None, version)