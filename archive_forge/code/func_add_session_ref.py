from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING, Final, MutableMapping
from weakref import WeakKeyDictionary
from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.util import HASHLIB_KWARGS
def add_session_ref(self, session: AppSession, script_run_count: int) -> None:
    """Adds a reference to a AppSession that has referenced
            this Entry's message.

            Parameters
            ----------
            session : AppSession
            script_run_count : int
                The session's run count at the time of the call

            """
    prev_run_count = self._session_script_run_counts.get(session, 0)
    if script_run_count < prev_run_count:
        _LOGGER.error('New script_run_count (%s) is < prev_run_count (%s). This should never happen!' % (script_run_count, prev_run_count))
        script_run_count = prev_run_count
    self._session_script_run_counts[session] = script_run_count