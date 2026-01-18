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
def list_deleted_files(self) -> List[str]:
    """
        Returns a list of the files that are deleted in the working directory or
        index.

        Returns:
            `List[str]`: A list of files that have been deleted in the working
            directory or index.
        """
    try:
        git_status = run_subprocess('git status -s', self.local_dir).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)
    if len(git_status) == 0:
        return []
    modified_files_statuses = [status.strip() for status in git_status.split('\n')]
    deleted_files_statuses = [status for status in modified_files_statuses if 'D' in status.split()[0]]
    deleted_files = [status.split()[-1].strip() for status in deleted_files_statuses]
    return deleted_files