from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_extract_bytes(self):
    text = b'From ff643aae102d8870cac88e8f007e70f58f3a7363 Mon Sep 17 00:00:00 2001\nFrom: Jelmer Vernooij <jelmer@samba.org>\nDate: Thu, 15 Apr 2010 15:40:28 +0200\nSubject: [PATCH 1/2] Remove executable bit from prey.ico (triggers a warning).\n\n---\n pixmaps/prey.ico |  Bin 9662 -> 9662 bytes\n 1 files changed, 0 insertions(+), 0 deletions(-)\n mode change 100755 => 100644 pixmaps/prey.ico\n\n-- \n1.7.0.4\n'
    c, diff, version = git_am_patch_split(BytesIO(text))
    self.assertEqual(b'Jelmer Vernooij <jelmer@samba.org>', c.committer)
    self.assertEqual(b'Jelmer Vernooij <jelmer@samba.org>', c.author)
    self.assertEqual(b'Remove executable bit from prey.ico (triggers a warning).\n', c.message)
    self.assertEqual(b' pixmaps/prey.ico |  Bin 9662 -> 9662 bytes\n 1 files changed, 0 insertions(+), 0 deletions(-)\n mode change 100755 => 100644 pixmaps/prey.ico\n\n', diff)
    self.assertEqual(b'1.7.0.4', version)