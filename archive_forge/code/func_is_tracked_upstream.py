import atexit
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TypedDict, Union
from urllib.parse import urlparse
from huggingface_hub.constants import REPO_TYPES_URL_PREFIXES, REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save
from .hf_api import HfApi, repo_type_and_id_from_hf_id
from .lfs import LFS_MULTIPART_UPLOAD_COMMAND
from .utils import (
from .utils._deprecation import _deprecate_method
def is_tracked_upstream(folder: Union[str, Path]) -> bool:
    """
    Check if the current checked-out branch is tracked upstream.

    Args:
        folder (`str` or `Path`):
            The folder in which to run the command.

    Returns:
        `bool`: `True` if the current checked-out branch is tracked upstream,
        `False` otherwise.
    """
    try:
        run_subprocess('git rev-parse --symbolic-full-name --abbrev-ref @{u}', folder)
        return True
    except subprocess.CalledProcessError as exc:
        if 'HEAD' in exc.stderr:
            raise OSError('No branch checked out')
        return False