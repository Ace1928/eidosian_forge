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
def test_empty_revid(self):
    """With an empty revid, we get only size an number and revisions"""
    backing = self.get_transport()
    request = smart_repo.SmartServerRepositoryGatherStats(backing)
    repository = self.make_repository('.')
    repository.gather_stats()
    expected_body = b'revisions: 0\n'
    self.assertEqual(smart_req.SmartServerResponse((b'ok',), expected_body), request.execute(b'', b'', b'no'))