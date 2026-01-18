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
def make_remote_repository(self, path, shared=None):
    """Make a RemoteRepository object backed by a real repository that will
        be created at the given path."""
    repo = self.make_repository(path, shared=shared)
    smart_server = test_server.SmartTCPServer_for_testing()
    self.start_server(smart_server, self.get_server())
    remote_transport = transport.get_transport_from_url(smart_server.get_url()).clone(path)
    if not repo.controldir._format.supports_transport(remote_transport):
        raise tests.TestNotApplicable('format does not support transport')
    remote_bzrdir = controldir.ControlDir.open_from_transport(remote_transport)
    remote_repo = remote_bzrdir.open_repository()
    return remote_repo