import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
def test_allow_diverged(self):
    """If allow_diverged is passed, then setting a divergent history
        succeeds.
        """
    self.make_branch_with_divergent_history()
    branch_token, repo_token = self.lock_branch()
    response = self.request.execute(b'', branch_token, repo_token, b'child-1', 1, 0)
    self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok', 2, b'child-1')), response)
    self.unlock_branch()
    self.assertEqual(b'child-1', self.tree.branch.last_revision())