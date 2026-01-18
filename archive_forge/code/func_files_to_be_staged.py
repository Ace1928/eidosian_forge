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
def files_to_be_staged(pattern: str='.', folder: Union[str, Path, None]=None) -> List[str]:
    """
    Returns a list of filenames that are to be staged.

    Args:
        pattern (`str` or `Path`):
            The pattern of filenames to check. Put `.` to get all files.
        folder (`str` or `Path`):
            The folder in which to run the command.

    Returns:
        `List[str]`: List of files that are to be staged.
    """
    try:
        p = run_subprocess('git ls-files --exclude-standard -mo'.split() + [pattern], folder)
        if len(p.stdout.strip()):
            files = p.stdout.strip().split('\n')
        else:
            files = []
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)
    return files