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
def lfs_prune(self, recent=False):
    """
        git lfs prune

        Args:
            recent (`bool`, *optional*, defaults to `False`):
                Whether to prune files even if they were referenced by recent
                commits. See the following
                [link](https://github.com/git-lfs/git-lfs/blob/f3d43f0428a84fc4f1e5405b76b5a73ec2437e65/docs/man/git-lfs-prune.1.ronn#recent-files)
                for more information.
        """
    try:
        with _lfs_log_progress():
            result = run_subprocess(f'git lfs prune {('--recent' if recent else '')}', self.local_dir)
            logger.info(result.stdout)
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)