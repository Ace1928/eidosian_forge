from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def start_new_revision(self, revision, parents, parent_invs):
    """Init the metadata needed for get_parents_and_revision_for_entry().

        :param revision: a Revision object
        """
    self._current_rev_id = revision.revision_id
    self._rev_parents = parents
    self._rev_parent_invs = parent_invs
    config = None
    self._commit_builder = self.repo.get_commit_builder(self.repo, parents, config, timestamp=revision.timestamp, timezone=revision.timezone, committer=revision.committer, revprops=revision.properties, revision_id=revision.revision_id)