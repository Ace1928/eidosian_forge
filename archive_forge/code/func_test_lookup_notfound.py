import os
import stat
from dulwich.objects import Blob, Commit, Tree
from ...revision import Revision
from ...tests import TestCase, TestCaseInTempDir, UnavailableFeature
from ...transport import get_transport
from ..cache import (DictBzrGitCache, IndexBzrGitCache, IndexGitCacheFormat,
def test_lookup_notfound(self):
    self.assertRaises(KeyError, list, self.map.lookup_git_sha(b'5686645d49063c73d35436192dfc9a160c672301'))