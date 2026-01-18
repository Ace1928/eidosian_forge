from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_fetch_order_BA(self):
    """See do_fetch_order_test"""
    self.do_fetch_order_test(b'B', b'A')