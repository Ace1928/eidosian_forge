from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def check_append_revisions_only(self, expected_value, value=None):
    """Set append_revisions_only in config and check its interpretation."""
    if value is not None:
        self.config_stack.set('append_revisions_only', value)
    self.assertEqual(expected_value, self.branch.get_append_revisions_only())