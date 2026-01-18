from __future__ import annotations
import collections
import threading
from typing import Final
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorage
def clear_session_refs(self, session_id: str | None=None) -> None:
    """Remove the given session's file references.

        (This does not remove any files from the manager - you must call
        `remove_orphaned_files` for that.)

        Should be called whenever ScriptRunner completes and when a session ends.

        Safe to call from any thread.
        """
    if session_id is None:
        session_id = _get_session_id()
    _LOGGER.debug('Disconnecting files for session with ID %s', session_id)
    with self._lock:
        if session_id in self._files_by_session_and_coord:
            del self._files_by_session_and_coord[session_id]
    _LOGGER.debug('Sessions still active: %r', self._files_by_session_and_coord.keys())
    _LOGGER.debug('Files: %s; Sessions with files: %s', len(self._file_metadata), len(self._files_by_session_and_coord))