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
def test_add_signature_text(self):
    transport_path = 'quack'
    repo, client = self.setup_fake_client_and_repository(transport_path)
    client.add_expected_call(b'Repository.lock_write', (b'quack/', b''), b'success', (b'ok', b'a token'))
    client.add_expected_call(b'Repository.start_write_group', (b'quack/', b'a token'), b'success', (b'ok', (b'token1',)))
    client.add_expected_call(b'Repository.add_signature_text', (b'quack/', b'a token', b'rev1', b'token1'), b'success', (b'ok',), None)
    repo.lock_write()
    repo.start_write_group()
    self.assertIs(None, repo.add_signature_text(b'rev1', b'every bloody emperor'))
    self.assertEqual(('call_with_body_bytes_expecting_body', b'Repository.add_signature_text', (b'quack/', b'a token', b'rev1', b'token1'), b'every bloody emperor'), client._calls[-1])