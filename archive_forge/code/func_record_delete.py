from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def record_delete(self, path, ie):
    self._add_entry((path, None, ie.file_id, None))
    self._paths_deleted_this_commit.add(path)
    if ie.kind == 'directory':
        try:
            del self.directory_entries[path]
        except KeyError:
            pass
        if self.basis_inventory.get_entry(ie.file_id).kind == 'directory':
            for child_relpath, entry in self.basis_inventory.iter_entries_by_dir(from_dir=ie.file_id):
                child_path = osutils.pathjoin(path, child_relpath)
                self._add_entry((child_path, None, entry.file_id, None))
                self._paths_deleted_this_commit.add(child_path)
                if entry.kind == 'directory':
                    try:
                        del self.directory_entries[child_path]
                    except KeyError:
                        pass