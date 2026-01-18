from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def bzr_file_id_and_new(self, path):
    """Get a Bazaar file identifier and new flag for a path.

        :return: file_id, is_new where
          is_new = True if the file_id is newly created
        """
    if path not in self._paths_deleted_this_commit:
        id = self._modified_file_ids.get(path)
        if id is not None:
            return (id, False)
        id = self.basis_inventory.path2id(path)
        if id is not None:
            return (id, False)
        if len(self.parents) > 1:
            for inv in self.parent_invs[1:]:
                id = self.basis_inventory.path2id(path)
                if id is not None:
                    return (id, False)
    dirname, basename = osutils.split(path)
    id = generate_ids.gen_file_id(basename)
    self.debug("Generated new file id %s for '%s' in revision-id '%s'", id, path, self.revision_id)
    self._new_file_ids[path] = id
    return (id, True)