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
def test_already_open_repo_and_reused_medium(self):
    """Bug 726584: create_branch(..., repository=repo) should work
        regardless of what the smart medium's base URL is.
        """
    self.transport_server = test_server.SmartTCPServer_for_testing
    transport = self.get_transport('.')
    repo = self.make_repository('quack')
    client = FakeClient(transport.base)
    transport = transport.clone('quack')
    reference_bzrdir_format = controldir.format_registry.get('default')()
    reference_format = reference_bzrdir_format.get_branch_format()
    network_name = reference_format.network_name()
    reference_repo_fmt = reference_bzrdir_format.repository_format
    reference_repo_name = reference_repo_fmt.network_name()
    client.add_expected_call(b'BzrDir.create_branch', (b'extra/quack/', network_name), b'success', (b'ok', network_name, b'', b'no', b'no', b'yes', reference_repo_name))
    a_controldir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
    branch = a_controldir.create_branch(repository=repo)
    self.assertIsInstance(branch, remote.RemoteBranch)
    format = branch._format
    self.assertEqual(network_name, format.network_name())