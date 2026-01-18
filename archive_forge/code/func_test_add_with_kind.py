from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_add_with_kind(self):
    branch = self.make_branch('branch')
    tree = branch.create_memorytree()
    tree.lock_write()
    tree.add(['', 'afile', 'adir'], ['directory', 'file', 'directory'])
    self.assertTrue(tree.is_versioned('afile'))
    self.assertFalse(tree.is_versioned('adir'))
    self.assertFalse(tree.has_filename('afile'))
    self.assertFalse(tree.has_filename('adir'))
    tree.unlock()