from ... import tests
from ...transport import memory
def test_cat_nonexistent_branch(self):
    self.vfs_transport_factory = memory.MemoryServer
    self.run_bzr_error(['^brz: ERROR: Not a branch'], ['cat', self.get_url()])