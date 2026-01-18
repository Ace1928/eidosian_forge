import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_get_revisions(self):
    tree = self.make_branch_and_tree('.')
    a_rev = tree.commit('initial empty commit', allow_pointless=True)
    b_rev = tree.commit('second empty commit', allow_pointless=True)
    c_rev = tree.commit('third empty commit', allow_pointless=True)
    repo = tree.branch.repository
    revision_ids = [a_rev, b_rev, c_rev]
    revisions = repo.get_revisions(revision_ids)
    self.assertEqual(len(revisions), 3)
    zipped = list(zip(revisions, revision_ids))
    self.assertEqual(len(zipped), 3)
    for revision, revision_id in zipped:
        self.assertEqual(revision.revision_id, revision_id)
        self.assertEqual(revision, repo.get_revision(revision_id))