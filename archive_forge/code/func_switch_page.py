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
def switch_page(self, page_path: str) -> AppTest:
    """Switch to another page of the app.

        This method does not automatically rerun the app. Use a follow-up call
        to ``AppTest.run()`` to obtain the elements on the selected page.

        Parameters
        ----------
        page_path: str
            Path of the page to switch to. The path must be relative to the
            main script's location (e.g. ``"pages/my_page.py"``).

        Returns
        -------
        AppTest
            self

        """
    main_dir = Path(self._script_path).parent
    full_page_path = main_dir / page_path
    if not full_page_path.is_file():
        raise ValueError(f'Unable to find script at {page_path}, make sure the page given is relative to the main script.')
    page_path_str = str(full_page_path.resolve())
    self._page_hash = calc_md5(page_path_str)
    return self