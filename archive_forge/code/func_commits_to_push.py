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
def commits_to_push(folder: Union[str, Path], upstream: Optional[str]=None) -> int:
    """
        Check the number of commits that would be pushed upstream

        Args:
            folder (`str` or `Path`):
                The folder in which to run the command.
            upstream (`str`, *optional*):
    The name of the upstream repository with which the comparison should be
    made.

        Returns:
            `int`: Number of commits that would be pushed upstream were a `git
            push` to proceed.
    """
    try:
        result = run_subprocess(f'git cherry -v {upstream or ''}', folder)
        return len(result.stdout.split('\n')) - 1
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)