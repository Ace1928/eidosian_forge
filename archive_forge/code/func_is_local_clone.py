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
def is_local_clone(folder: Union[str, Path], remote_url: str) -> bool:
    """
    Check if the folder is a local clone of the remote_url

    Args:
        folder (`str` or `Path`):
            The folder in which to run the command.
        remote_url (`str`):
            The url of a git repository.

    Returns:
        `bool`: `True` if the repository is a local clone of the remote
        repository specified, `False` otherwise.
    """
    if not is_git_repo(folder):
        return False
    remotes = run_subprocess('git remote -v', folder).stdout
    remote_url = re.sub('https://.*@', 'https://', remote_url)
    remotes = [re.sub('https://.*@', 'https://', remote) for remote in remotes.split()]
    return remote_url in remotes