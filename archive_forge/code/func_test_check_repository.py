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
def test_check_repository(self):
    """Check a fairly simple repository's history"""
    tree = self.make_branch_and_tree('.')
    a_rev = tree.commit('initial empty commit', allow_pointless=True)
    result = tree.branch.repository.check()
    result.report_results(verbose=True)
    result.report_results(verbose=False)