import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
def test_initialized_dir(self):
    """Initializing an extant directory should fail like the bzrdir api."""
    backing = self.get_transport()
    name = self.make_controldir('reference')._format.network_name()
    request = smart_dir.SmartServerRequestBzrDirInitializeEx(backing)
    self.make_controldir('subdir')
    self.assertRaises(transport.FileExists, request.execute, name, b'subdir', b'False', b'False', b'False', b'', b'', b'', b'', b'False')