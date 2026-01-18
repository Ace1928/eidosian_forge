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
def is_tracked_with_lfs(filename: Union[str, Path]) -> bool:
    """
    Check if the file passed is tracked with git-lfs.

    Args:
        filename (`str` or `Path`):
            The filename to check.

    Returns:
        `bool`: `True` if the file passed is tracked with git-lfs, `False`
        otherwise.
    """
    folder = Path(filename).parent
    filename = Path(filename).name
    try:
        p = run_subprocess('git check-attr -a'.split() + [filename], folder)
        attributes = p.stdout.strip()
    except subprocess.CalledProcessError as exc:
        if not is_git_repo(folder):
            return False
        else:
            raise OSError(exc.stderr)
    if len(attributes) == 0:
        return False
    found_lfs_tag = {'diff': False, 'merge': False, 'filter': False}
    for attribute in attributes.split('\n'):
        for tag in found_lfs_tag.keys():
            if tag in attribute and 'lfs' in attribute:
                found_lfs_tag[tag] = True
    return all(found_lfs_tag.values())