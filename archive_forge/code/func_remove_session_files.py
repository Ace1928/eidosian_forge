from __future__ import annotations
import io
from abc import abstractmethod
from typing import NamedTuple, Protocol, Sequence
from streamlit import util
from streamlit.proto.Common_pb2 import FileURLs as FileURLsProto
from streamlit.runtime.stats import CacheStatsProvider
@abstractmethod
def remove_session_files(self, session_id: str) -> None:
    """Remove all files associated with a given session."""
    raise NotImplementedError