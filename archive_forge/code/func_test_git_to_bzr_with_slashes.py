from .... import tests
from .. import branch_mapper
from . import FastimportFeature
def test_git_to_bzr_with_slashes(self):
    m = branch_mapper.BranchMapper()
    for git, bzr in {b'refs/heads/master/slave': 'master/slave', b'refs/heads/foo/bar': 'foo/bar', b'refs/tags/master/slave': 'master/slave.tag', b'refs/tags/foo/bar': 'foo/bar.tag', b'refs/remotes/origin/master/slave': 'master/slave.remote', b'refs/remotes/origin/foo/bar': 'foo/bar.remote'}.items():
        self.assertEqual(m.git_to_bzr(git), bzr)