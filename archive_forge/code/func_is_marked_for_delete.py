from __future__ import annotations
import collections
import threading
from typing import Final
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorage
@property
def is_marked_for_delete(self) -> bool:
    return self._is_marked_for_delete