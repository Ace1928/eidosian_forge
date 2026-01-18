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
def git_config_username_and_email(self, git_user: Optional[str]=None, git_email: Optional[str]=None):
    """
        Sets git username and email (only in the current repo).

        Args:
            git_user (`str`, *optional*):
                The username to register through `git`.
            git_email (`str`, *optional*):
                The email to register through `git`.
        """
    try:
        if git_user is not None:
            run_subprocess('git config user.name'.split() + [git_user], self.local_dir)
        if git_email is not None:
            run_subprocess(f'git config user.email {git_email}'.split(), self.local_dir)
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)