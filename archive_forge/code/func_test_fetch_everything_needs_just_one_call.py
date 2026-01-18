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
def test_fetch_everything_needs_just_one_call(self):
    local = self.make_branch('local')
    builder = self.make_branch_builder('remote')
    builder.build_commit(message='Commit.')
    remote_branch_url = self.smart_server.get_url() + 'remote'
    remote_branch = bzrdir.BzrDir.open(remote_branch_url).open_branch()
    self.hpss_calls = []
    local.repository.fetch(remote_branch.repository, fetch_spec=vf_search.EverythingResult(remote_branch.repository))
    self.assertEqual([b'Repository.get_stream_1.19'], self.hpss_calls)