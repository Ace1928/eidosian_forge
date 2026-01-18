from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def copy_handler(self, filecmd):
    src_path = self._decode_path(filecmd.src_path)
    dest_path = self._decode_path(filecmd.dest_path)
    self.debug('copying %s to %s', src_path, dest_path)
    self._copy_item(src_path, dest_path, self.basis_inventory)