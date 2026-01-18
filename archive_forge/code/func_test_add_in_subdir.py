from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_add_in_subdir(self):
    branch = self.make_branch('branch')
    tree = branch.create_memorytree()
    with tree.lock_write():
        tree.add([''], ['directory'])
        tree.mkdir('adir')
        tree.put_file_bytes_non_atomic('adir/afile', b'barshoom')
        tree.add(['adir/afile'], ['file'])
        self.assertTrue(tree.is_versioned('adir/afile'))
        self.assertTrue(tree.is_versioned('adir'))