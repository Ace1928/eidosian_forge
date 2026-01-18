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
def test_file_in_unchanged_directory(self):
    delta = list(self.repository.get_revision_deltas([self.repository.get_revision(self.rev2)], specific_files=['bar/b3']))[0]
    self.assertIsInstance(delta, _mod_delta.TreeDelta)
    if [(c.path[1], c.kind[1]) for c in delta.added] == [('bar', 'directory'), ('bar/b3', 'file')]:
        self.knownFailure("bzr incorrectly reports 'bar' as added - bug 878217")
    self.assertEqual([('bar/b3', 'file')], [(c.path[1], c.kind[1]) for c in delta.added])