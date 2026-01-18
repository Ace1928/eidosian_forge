from __future__ import annotations
import math
import os
import shutil
from typing import Final
from streamlit import util
from streamlit.file_util import get_streamlit_file_path, streamlit_read, streamlit_write
from streamlit.logger import get_logger
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import (
def get_cache_folder_path() -> str:
    return get_streamlit_file_path(_CACHE_DIR_NAME)