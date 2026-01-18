from __future__ import annotations
from typing import Callable, Final, List, cast
from streamlit.logger import get_logger
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.session_manager import (
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.watcher import LocalSourcesWatcher
def disconnect_session(self, session_id: str) -> None:
    if session_id in self._active_session_info_by_id:
        active_session_info = self._active_session_info_by_id[session_id]
        session = active_session_info.session
        session.request_script_stop()
        session.disconnect_file_watchers()
        self._session_storage.save(SessionInfo(client=None, session=session, script_run_count=active_session_info.script_run_count))
        del self._active_session_info_by_id[session_id]