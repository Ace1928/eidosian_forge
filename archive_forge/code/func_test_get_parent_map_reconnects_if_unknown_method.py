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
def test_get_parent_map_reconnects_if_unknown_method(self):
    transport_path = 'quack'
    rev_id = b'revision-id'
    repo, client = self.setup_fake_client_and_repository(transport_path)
    client.add_unknown_method_response(b'Repository.get_parent_map')
    client.add_success_response_with_body(rev_id, b'ok')
    self.assertFalse(client._medium._is_remote_before((1, 2)))
    parents = repo.get_parent_map([rev_id])
    self.assertEqual([('call_with_body_bytes_expecting_body', b'Repository.get_parent_map', (b'quack/', b'include-missing:', rev_id), b'\n\n0'), ('disconnect medium',), ('call_expecting_body', b'Repository.get_revision_graph', (b'quack/', b''))], client._calls)
    self.assertTrue(client._medium._is_remote_before((1, 2)))
    self.assertEqual({rev_id: (b'null:',)}, parents)