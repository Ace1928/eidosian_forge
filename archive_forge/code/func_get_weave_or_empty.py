import os
from .... import errors, osutils
from .... import transport as _mod_transport
from .... import ui
from ....trace import mutter
from . import TransportStore
def get_weave_or_empty(self, file_id, transaction):
    """Return a weave, or an empty one if it doesn't exist."""
    _filename = self.filename(file_id)
    try:
        return self.get_weave(file_id, transaction, _filename=_filename)
    except _mod_transport.NoSuchFile:
        weave = self._make_new_versionedfile(file_id, transaction, known_missing=True, _filename=_filename)
        return weave