import sys
from io import BytesIO
from stat import S_ISDIR
from ...bzr.bzrdir import BzrDirMetaFormat1
from ...bzr.serializer import format_registry as serializer_format_registry
from ...errors import IllegalPath
from ...repository import InterRepository, Repository
from ...tests import TestCase, TestCaseWithTransport
from ...transport import NoSuchFile
from . import xml4
from .bzrdir import BzrDirFormat6
from .repository import (InterWeaveRepo, RepositoryFormat4, RepositoryFormat5,
def test_disk_layout(self):
    control = BzrDirMetaFormat1().initialize(self.get_url())
    repo = RepositoryFormat7().initialize(control)
    repo.lock_write()
    repo.unlock()
    t = control.get_repository_transport(None)
    with t.get('format') as f:
        self.assertEqualDiff(b'Bazaar-NG Repository format 7', f.read())
    self.assertTrue(S_ISDIR(t.stat('revision-store').st_mode))
    self.assertTrue(S_ISDIR(t.stat('weaves').st_mode))
    with t.get('inventory.weave') as f:
        self.assertEqualDiff(b'# bzr weave file v5\nw\nW\n', f.read())
    control.create_branch()
    tree = control.create_workingtree()
    tree.add(['foo'], ['file'], ids=[b'Foo:Bar'])
    tree.put_file_bytes_non_atomic('foo', b'content\n')
    try:
        tree.commit('first post', rev_id=b'first')
    except IllegalPath:
        if sys.platform != 'win32':
            raise
        self.knownFailure('Foo:Bar cannot be used as a file-id on windows in repo format 7')
        return
    with t.get('weaves/74/Foo%3ABar.weave') as f:
        self.assertEqualDiff(b'# bzr weave file v5\ni\n1 7fe70820e08a1aac0ef224d9c66ab66831cc4ab1\nn first\n\nw\n{ 0\n. content\n}\nW\n', f.read())