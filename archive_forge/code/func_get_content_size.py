from __future__ import annotations
from urllib.parse import quote
import tornado.web
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorageError
from streamlit.runtime.memory_media_file_storage import (
from streamlit.web.server import allow_cross_origin_requests
def get_content_size(self) -> int:
    abspath = self.absolute_path
    if abspath is None:
        return 0
    media_file = self._storage.get_file(abspath)
    return media_file.content_size