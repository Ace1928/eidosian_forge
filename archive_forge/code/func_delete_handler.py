from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def delete_handler(self, filecmd):
    self.debug('deleting %s', filecmd.path)
    self._delete_item(self._decode_path(filecmd.path), self.basis_inventory)