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
def test_get_parent_map_negative_caches_missing_keys(self):
    self.setup_smart_server_with_call_log()
    repo = self.make_repository('foo')
    self.assertIsInstance(repo, RemoteRepository)
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.reset_smart_call_log()
    graph = repo.get_graph()
    self.assertEqual({}, graph.get_parent_map([b'some-missing', b'other-missing']))
    self.assertLength(1, self.hpss_calls)
    self.reset_smart_call_log()
    graph = repo.get_graph()
    self.assertEqual({}, graph.get_parent_map([b'some-missing', b'other-missing']))
    self.assertLength(0, self.hpss_calls)
    self.reset_smart_call_log()
    graph = repo.get_graph()
    self.assertEqual({}, graph.get_parent_map([b'some-missing', b'other-missing', b'more-missing']))
    self.assertLength(1, self.hpss_calls)