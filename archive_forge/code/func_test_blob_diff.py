from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_blob_diff(self):
    f = BytesIO()
    write_blob_diff(f, (b'foo.txt', 420, Blob.from_string(b'old\nsame\n')), (b'bar.txt', 420, Blob.from_string(b'new\nsame\n')))
    self.assertEqual([b'diff --git a/foo.txt b/bar.txt', b'index 3b0f961..a116b51 644', b'--- a/foo.txt', b'+++ b/bar.txt', b'@@ -1,2 +1,2 @@', b'-old', b'+new', b' same'], f.getvalue().splitlines())