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
def inventory_delta_substream():
    entry = inv.make_entry('directory', 'newdir', inv.root.file_id, b'newdir-id')
    entry.revision = b'ghost'
    delta = [(None, 'newdir', b'newdir-id', entry)]
    serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=False)
    lines = serializer.delta_to_lines(b'rev1', b'rev2', delta)
    yield versionedfile.ChunkedContentFactory((b'rev2',), (b'rev1',), None, lines)
    lines = serializer.delta_to_lines(b'rev1', b'rev3', delta)
    yield versionedfile.ChunkedContentFactory((b'rev3',), (b'rev1',), None, lines)