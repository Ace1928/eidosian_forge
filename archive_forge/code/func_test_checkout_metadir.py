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
def test_checkout_metadir(self):
    backing = self.get_transport()
    request = smart_dir.SmartServerBzrDirRequestCheckoutMetaDir(backing)
    self.make_branch('.', format='2a')
    response = request.execute(b'')
    self.assertEqual(smart_req.SmartServerResponse((b'Bazaar-NG meta directory, format 1\n', b'Bazaar repository format 2a (needs bzr 1.16 or later)\n', b'Bazaar Branch Format 7 (needs bzr 1.6)\n')), response)