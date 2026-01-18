from __future__ import annotations
import contextlib
import errno
import io
import os
from pathlib import Path
from streamlit import env_util, util
from streamlit.string_util import is_binary_string
def get_streamlit_file_path(*filepath) -> str:
    """Return the full path to a file in ~/.streamlit.

    This doesn't guarantee that the file (or its directory) exists.
    """
    home = os.path.expanduser('~')
    if home is None:
        raise RuntimeError('No home directory.')
    return os.path.join(home, CONFIG_FOLDER_NAME, *filepath)