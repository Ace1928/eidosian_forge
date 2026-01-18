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
@property
def current_branch(self) -> str:
    """
        Returns the current checked out branch.

        Returns:
            `str`: Current checked out branch.
        """
    try:
        result = run_subprocess('git rev-parse --abbrev-ref HEAD', self.local_dir).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)
    return result