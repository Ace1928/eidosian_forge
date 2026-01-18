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
def test_exposes_get_cached_parent_map(self):
    """RemoteRepository exposes get_cached_parent_map from
        _unstacked_provider
        """
    r1 = 'ำ'.encode()
    r2 = 'ණ'.encode()
    lines = [b' '.join([r2, r1]), r1]
    encoded_body = bz2.compress(b'\n'.join(lines))
    transport_path = 'quack'
    repo, client = self.setup_fake_client_and_repository(transport_path)
    client.add_success_response_with_body(encoded_body, b'ok')
    repo.lock_read()
    self.assertEqual({}, repo.get_cached_parent_map([r1]))
    self.assertEqual([], client._calls)
    self.assertEqual({r2: (r1,)}, repo.get_parent_map([r2]))
    self.assertEqual({r1: (NULL_REVISION,)}, repo.get_cached_parent_map([r1]))
    self.assertEqual([('call_with_body_bytes_expecting_body', b'Repository.get_parent_map', (b'quack/', b'include-missing:', r2), b'\n\n0')], client._calls)
    repo.unlock()