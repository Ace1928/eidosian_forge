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
def is_git_ignored(filename: Union[str, Path]) -> bool:
    """
    Check if file is git-ignored. Supports nested .gitignore files.

    Args:
        filename (`str` or `Path`):
            The filename to check.

    Returns:
        `bool`: `True` if the file passed is ignored by `git`, `False`
        otherwise.
    """
    folder = Path(filename).parent
    filename = Path(filename).name
    try:
        p = run_subprocess('git check-ignore'.split() + [filename], folder, check=False)
        is_ignored = not bool(p.returncode)
    except subprocess.CalledProcessError as exc:
        raise OSError(exc.stderr)
    return is_ignored