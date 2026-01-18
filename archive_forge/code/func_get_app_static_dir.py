from __future__ import annotations
import contextlib
import errno
import io
import os
from pathlib import Path
from streamlit import env_util, util
from streamlit.string_util import is_binary_string
def get_app_static_dir(main_script_path: str) -> str:
    """Get the folder where app static files live"""
    main_script_path = Path(main_script_path)
    static_dir = main_script_path.parent / APP_STATIC_FOLDER_NAME
    return os.path.abspath(static_dir)