import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
def iter_rev_trees(self, revs):
    """Iterate through RevisionTrees efficiently.

        Additionally, the inventory's revision_id is set if unset.

        Trees are retrieved in batches of 100, and then yielded in the order
        they were requested.

        Args:
          revs: A list of revision ids
        """
    revs = list(revs)
    while revs:
        for tree in self.source.revision_trees(revs[:100]):
            if tree.root_inventory.revision_id is None:
                tree.root_inventory.revision_id = tree.get_revision_id()
            yield tree
        revs = revs[100:]