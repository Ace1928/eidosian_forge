import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
def test_hpss_get_single_revision(self):
    transport_path = 'quack'
    repo, client = self.setup_fake_client_and_repository(transport_path)
    somerev1 = Revision(b'somerev1')
    somerev1.committer = 'Joe Committer <joe@example.com>'
    somerev1.timestamp = 1321828927
    somerev1.timezone = -60
    somerev1.inventory_sha1 = b'691b39be74c67b1212a75fcb19c433aaed903c2b'
    somerev1.message = 'Message'
    body = zlib.compress(b''.join(chk_bencode_serializer.write_revision_to_lines(somerev1)))
    client.add_success_response_with_body([body[:10], body[10:]], b'ok', b'10')
    revs = repo.get_revisions([b'somerev1'])
    self.assertEqual(revs, [somerev1])
    self.assertEqual([('call_with_body_bytes_expecting_body', b'Repository.iter_revisions', (b'quack/',), b'somerev1')], client._calls)