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
def test_attribute_format_supports_overriding_transport(self):
    repo = self.make_repository('repo')
    self.assertIn(repo._format.supports_overriding_transport, (True, False))
    repo.control_transport.copy_tree('.', '../repository.backup')
    backup_transport = repo.control_transport.clone('../repository.backup')
    if repo._format.supports_overriding_transport:
        backup = repo._format.open(repo.controldir, _override_transport=backup_transport)
        self.assertIs(backup_transport, backup.control_transport)
    else:
        self.assertRaises(TypeError, repo._format.open, repo.controldir, _override_transport=backup_transport)