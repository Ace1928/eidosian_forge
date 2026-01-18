from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING, Final, MutableMapping
from weakref import WeakKeyDictionary
from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.util import HASHLIB_KWARGS
def has_session_ref(self, session: AppSession) -> bool:
    return session in self._session_script_run_counts