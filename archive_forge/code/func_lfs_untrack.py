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
def lfs_untrack(self, patterns: Union[str, List[str]]):
    """
        Tell git-lfs to untrack those files.

        Args:
            patterns (`Union[str, List[str]]`):
                The pattern, or list of patterns, to untrack with git-lfs.
        """
    if isinstance(patterns, str):
        patterns = [patterns]
    try:
        for pattern in patterns:
            run_subprocess('git lfs untrack'.split() + [pattern], self.local_dir)
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)