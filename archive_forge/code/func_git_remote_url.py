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
def git_remote_url(self) -> str:
    """
        Get URL to origin remote.

        Returns:
            `str`: The URL of the `origin` remote.
        """
    try:
        p = run_subprocess('git config --get remote.origin.url', self.local_dir)
        url = p.stdout.strip()
        return re.sub('https://.*@', 'https://', url)
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)