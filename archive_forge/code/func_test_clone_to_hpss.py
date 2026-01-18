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
def test_clone_to_hpss(self):
    if not self.repository_format.supports_leaving_lock:
        raise tests.TestNotApplicable('Cannot lock pre_metadir_formats remotely.')
    remote_transport = self.make_smart_server('remote')
    local_branch = self.make_branch('local')
    remote_branch = local_branch.create_clone_on_transport(remote_transport)
    self.assertEqual(local_branch.repository._format.supports_external_lookups, remote_branch.repository._format.supports_external_lookups)