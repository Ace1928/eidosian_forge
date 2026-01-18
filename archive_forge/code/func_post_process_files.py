from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def post_process_files(self):
    """Save the revision."""
    delta = self._get_final_delta()
    inv = self.rev_store.load_using_delta(self.revision, self.basis_inventory, delta, None, self._get_data, self._get_per_file_parents, self._get_inventories)
    self.cache_mgr.inventories[self.revision_id] = inv