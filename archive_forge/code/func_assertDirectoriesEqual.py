import errno
from stat import S_ISDIR
import breezy.branch
from breezy import controldir, errors, repository
from breezy import revision as _mod_revision
from breezy import transport, workingtree
from breezy.bzr import bzrdir
from breezy.bzr.remote import RemoteBzrDirFormat
from breezy.bzr.tests.per_bzrdir import TestCaseWithBzrDir
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.transport import FileExists
from breezy.transport.local import LocalTransport
def assertDirectoriesEqual(self, source, target, ignore_list=[]):
    """Assert that the content of source and target are identical.

        paths in ignore list will be completely ignored.

        We ignore paths that represent data which is allowed to change during
        a clone or sprout: for instance, inventory.knit contains gzip fragements
        which have timestamps in them, and as we have read the inventory from
        the source knit, the already-read data is recompressed rather than
        reading it again, which leads to changed timestamps. This is ok though,
        because the inventory.kndx file is not ignored, and the integrity of
        knit joins is tested by test_knit and test_versionedfile.

        :seealso: Additionally, assertRepositoryHasSameItems provides value
            rather than representation checking of repositories for
            equivalence.
        """
    files = []
    directories = ['.']
    while directories:
        dir = directories.pop()
        for path in set(source.list_dir(dir) + target.list_dir(dir)):
            path = dir + '/' + path
            if path in ignore_list:
                continue
            try:
                stat = source.stat(path)
            except transport.NoSuchFile:
                self.fail('%s not in source' % path)
            if S_ISDIR(stat.st_mode):
                self.assertTrue(S_ISDIR(target.stat(path).st_mode))
                directories.append(path)
            else:
                self.assertEqualDiff(source.get_bytes(path), target.get_bytes(path), 'text for file %r differs:\n' % path)