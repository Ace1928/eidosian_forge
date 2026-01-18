from .object_store import iter_tree_contents
from .objects import Blob
from .patch import is_binary
def checkin_normalize(self, blob, tree_path):
    if self.fallback_read_filter is not None or tree_path not in self.existing_paths:
        return super().checkin_normalize(blob, tree_path)
    return blob