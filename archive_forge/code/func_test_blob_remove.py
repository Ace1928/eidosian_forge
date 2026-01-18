from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_blob_remove(self):
    f = BytesIO()
    write_blob_diff(f, (b'bar.txt', 420, Blob.from_string(b'new\nsame\n')), (None, None, None))
    self.assertEqual([b'diff --git a/bar.txt b/bar.txt', b'deleted file mode 644', b'index a116b51..0000000', b'--- a/bar.txt', b'+++ /dev/null', b'@@ -1,2 +0,0 @@', b'-new', b'-same'], f.getvalue().splitlines())