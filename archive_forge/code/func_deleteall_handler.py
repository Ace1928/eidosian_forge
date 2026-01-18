from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def deleteall_handler(self, filecmd):
    self.debug('deleting all files (and also all directories)')
    self._delete_all_items(self.basis_inventory)