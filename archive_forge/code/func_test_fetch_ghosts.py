from .. import TestCaseWithTransport
def test_fetch_ghosts(self):
    self.run_bzr('init')
    self.run_bzr('fetch-ghosts .')