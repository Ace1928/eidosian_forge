import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_gather_related_braches(self):
    branch = self.make_branch('.')
    branch.lock_write()
    try:
        branch.set_public_branch('baz')
        branch.set_push_location('bar')
        branch.set_parent('foo')
        branch.set_submit_branch('qux')
    finally:
        branch.unlock()
    self.assertEqual([('public branch', 'baz'), ('push branch', 'bar'), ('parent branch', 'foo'), ('submit branch', 'qux')], info._gather_related_branches(branch).locs)