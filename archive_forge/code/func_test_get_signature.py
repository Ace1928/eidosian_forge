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
def test_get_signature(self):
    backing = self.get_transport()
    request = smart_repo.SmartServerRepositoryGetRevisionSignatureText(backing)
    bb = self.make_branch_builder('.')
    bb.build_commit(rev_id=b'A')
    repo = bb.get_branch().repository
    strategy = gpg.LoopbackGPGStrategy(None)
    self.addCleanup(repo.lock_write().unlock)
    repo.start_write_group()
    repo.sign_revision(b'A', strategy)
    repo.commit_write_group()
    expected_body = b'-----BEGIN PSEUDO-SIGNED CONTENT-----\n' + Testament.from_revision(repo, b'A').as_short_text() + b'-----END PSEUDO-SIGNED CONTENT-----\n'
    self.assertEqual(smart_req.SmartServerResponse((b'ok',), expected_body), request.execute(b'', b'A'))