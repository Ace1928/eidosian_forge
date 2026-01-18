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
def test_create_bundle(self):
    wt = self.make_branch_and_tree('repo')
    self.build_tree(['repo/file1'])
    wt.add('file1')
    rev1 = wt.commit('file1')
    fileobj = BytesIO()
    wt.branch.repository.create_bundle(rev1, _mod_revision.NULL_REVISION, fileobj)