import os
from .... import errors, osutils
from .... import transport as _mod_transport
from .... import ui
from ....trace import mutter
from . import TransportStore
def get_weave(self, file_id, transaction, _filename=None):
    """Return the VersionedFile for file_id.

        :param _filename: filename that would be returned from self.filename for
        file_id. This is used to reduce duplicate filename calculations when
        using 'get_weave_or_empty'. FOR INTERNAL USE ONLY.
        """
    if _filename is None:
        _filename = self.filename(file_id)
    if transaction.writeable():
        w = self._versionedfile_class(_filename, self._transport, self._file_mode, get_scope=self.get_scope, **self._versionedfile_kwargs)
    else:
        w = self._versionedfile_class(_filename, self._transport, self._file_mode, create=False, access_mode='r', get_scope=self.get_scope, **self._versionedfile_kwargs)
    return w