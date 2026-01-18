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
def make_branch_with_tags(self):
    self.setup_smart_server_with_call_log()
    builder = self.make_branch_builder('foo')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', ''))], revision_id=b'tip')
    builder.finish_series()
    branch = builder.get_branch()
    branch.tags.set_tag('tag-1', b'rev-1')
    branch.tags.set_tag('tag-2', b'rev-2')
    return branch