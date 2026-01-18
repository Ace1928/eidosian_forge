from .. import ui
from ..branch import Branch
from ..check import Check
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import note
from ..workingtree import WorkingTree
def check_revisions(self):
    """Scan revisions, checking data directly available as we go."""
    revision_iterator = self.repository.iter_revisions(self.repository.all_revision_ids())
    revision_iterator = self._check_revisions(revision_iterator)
    if not self.repository._format.revision_graph_can_have_wrong_parents:
        self.revs_with_bad_parents_in_index = None
        for thing in revision_iterator:
            pass
    else:
        bad_revisions = self.repository._find_inconsistent_revision_parents(revision_iterator)
        self.revs_with_bad_parents_in_index = list(bad_revisions)