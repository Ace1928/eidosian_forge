from __future__ import annotations
import contextlib
import errno
import io
import os
from pathlib import Path
from streamlit import env_util, util
from streamlit.string_util import is_binary_string
def get_project_streamlit_file_path(*filepath):
    """Return the full path to a filepath in ${CWD}/.streamlit.

    This doesn't guarantee that the file (or its directory) exists.
    """
    return os.path.join(os.getcwd(), CONFIG_FOLDER_NAME, *filepath)