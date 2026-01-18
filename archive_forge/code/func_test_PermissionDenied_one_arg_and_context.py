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
def test_PermissionDenied_one_arg_and_context(self):
    """Given a choice between a path from the local context and a path on
        the wire, _translate_error prefers the path from the local context.
        """
    local_path = 'local path'
    remote_path = 'remote path'
    translated_error = self.translateTuple((b'PermissionDenied', remote_path.encode('utf-8')), path=local_path)
    expected_error = errors.PermissionDenied(local_path)
    self.assertEqual(expected_error, translated_error)