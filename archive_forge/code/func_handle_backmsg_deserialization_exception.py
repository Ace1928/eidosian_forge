from __future__ import annotations
import asyncio
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Final, NamedTuple
from streamlit import config
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.caching import (
from streamlit.runtime.caching.storage.local_disk_cache_storage import (
from streamlit.runtime.forward_msg_cache import (
from streamlit.runtime.legacy_caching.caching import _mem_caches
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.media_file_storage import MediaFileStorage
from streamlit.runtime.memory_session_storage import MemorySessionStorage
from streamlit.runtime.runtime_util import is_cacheable_msg
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.session_manager import (
from streamlit.runtime.state import (
from streamlit.runtime.stats import StatsManager
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.runtime.websocket_session_manager import WebsocketSessionManager
from streamlit.watcher import LocalSourcesWatcher
def handle_backmsg_deserialization_exception(self, session_id: str, exc: BaseException) -> None:
    """Handle an Exception raised during deserialization of a BackMsg.

        Parameters
        ----------
        session_id
            The session's unique ID.
        exc
            The Exception.

        Notes
        -----
        Threading: UNSAFE. Must be called on the eventloop thread.
        """
    if self._state in (RuntimeState.STOPPING, RuntimeState.STOPPED):
        raise RuntimeStoppedError(f"Can't handle_backmsg_deserialization_exception (state={self._state})")
    session_info = self._session_mgr.get_active_session_info(session_id)
    if session_info is None:
        _LOGGER.debug('Discarding BackMsg Exception for disconnected session (id=%s)', session_id)
        return
    session_info.session.handle_backmsg_exception(exc)