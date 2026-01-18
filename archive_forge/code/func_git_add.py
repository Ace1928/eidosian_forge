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
def git_add(self, pattern: str='.', auto_lfs_track: bool=False):
    """
        git add

        Setting the `auto_lfs_track` parameter to `True` will automatically
        track files that are larger than 10MB with `git-lfs`.

        Args:
            pattern (`str`, *optional*, defaults to "."):
                The pattern with which to add files to staging.
            auto_lfs_track (`bool`, *optional*, defaults to `False`):
                Whether to automatically track large and binary files with
                git-lfs. Any file over 10MB in size, or in binary format, will
                be automatically tracked.
        """
    if auto_lfs_track:
        tracked_files = self.auto_track_large_files(pattern)
        tracked_files.extend(self.auto_track_binary_files(pattern))
        if tracked_files:
            logger.warning(f'Adding files tracked by Git LFS: {tracked_files}. This may take a bit of time if the files are large.')
    try:
        result = run_subprocess('git add -v'.split() + [pattern], self.local_dir)
        logger.info(f'Adding to index:\n{result.stdout}\n')
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)