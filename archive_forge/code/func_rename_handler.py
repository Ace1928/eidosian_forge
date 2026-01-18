from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def rename_handler(self, filecmd):
    old_path = self._decode_path(filecmd.old_path)
    new_path = self._decode_path(filecmd.new_path)
    self.debug('renaming %s to %s', old_path, new_path)
    self._rename_item(old_path, new_path, self.basis_inventory)