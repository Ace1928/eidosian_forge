from io import StringIO
from .. import config
from .. import status as _mod_status
from ..revisionspec import RevisionSpec
from ..status import show_pending_merges, show_tree_status
from . import TestCaseWithTransport
def test_installed_hooks_are_StatusHooks(self):
    """The installed hooks object should be a StatusHooks.
        """
    self.assertIsInstance(self._preserved_hooks[_mod_status][1], _mod_status.StatusHooks)