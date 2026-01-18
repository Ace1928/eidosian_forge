from __future__ import annotations
import hashlib
import inspect
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence
from unittest.mock import MagicMock
from urllib import parse
from streamlit import source_util
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime import Runtime
from streamlit.runtime.caching.storage.dummy_cache_storage import (
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.runtime.secrets import Secrets
from streamlit.runtime.state.common import TESTING_KEY
from streamlit.runtime.state.safe_session_state import SafeSessionState
from streamlit.runtime.state.session_state import SessionState
from streamlit.testing.v1.element_tree import (
from streamlit.testing.v1.local_script_runner import LocalScriptRunner
from streamlit.testing.v1.util import patch_config_options
from streamlit.util import HASHLIB_KWARGS, calc_md5
@property
def subheader(self) -> ElementList[Subheader]:
    """Sequence of all ``st.subheader`` elements.

        Returns
        -------
        ElementList of Subheader
            Sequence of all ``st.subheader`` elements. Individual elements can be
            accessed from an ElementList by index (order on the page). For
            example, ``at.subheader[0]`` for the first element. Subheader is an
            extension of the Element class.
        """
    return self._tree.subheader