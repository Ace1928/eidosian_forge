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
def test_backwards_compatible_no_tags(self):
    br = self.make_branch_with_tags()
    br.get_config_stack().set('branch.fetch_tags', False)
    self.addCleanup(br.lock_read().unlock)
    verb = b'Branch.heads_to_fetch'
    self.disable_verb(verb)
    self.reset_smart_call_log()
    result = br.heads_to_fetch()
    self.assertEqual(({b'tip'}, set()), result)
    self.assertEqual([b'Branch.last_revision_info'], [call.call.method for call in self.hpss_calls])