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
def test_backwards_compat_set_option_with_dict(self):
    self.setup_smart_server_with_call_log()
    branch = self.make_branch('.')
    verb = b'Branch.set_config_option_dict'
    self.disable_verb(verb)
    branch.lock_write()
    self.addCleanup(branch.unlock)
    self.reset_smart_call_log()
    config = branch._get_config()
    value_dict = {'ascii': 'a', 'unicode ⌚': '‽'}
    config.set_option(value_dict, 'name')
    self.assertLength(11, self.hpss_calls)
    self.assertEqual(value_dict, branch._get_config().get_option('name'))