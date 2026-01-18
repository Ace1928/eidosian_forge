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
def git_push(self, upstream: Optional[str]=None, blocking: bool=True, auto_lfs_prune: bool=False) -> Union[str, Tuple[str, CommandInProgress]]:
    """
        git push

        If used without setting `blocking`, will return url to commit on remote
        repo. If used with `blocking=True`, will return a tuple containing the
        url to commit and the command object to follow for information about the
        process.

        Args:
            upstream (`str`, *optional*):
                Upstream to which this should push. If not specified, will push
                to the lastly defined upstream or to the default one (`origin
                main`).
            blocking (`bool`, *optional*, defaults to `True`):
                Whether the function should return only when the push has
                finished. Setting this to `False` will return an
                `CommandInProgress` object which has an `is_done` property. This
                property will be set to `True` when the push is finished.
            auto_lfs_prune (`bool`, *optional*, defaults to `False`):
                Whether to automatically prune files once they have been pushed
                to the remote.
        """
    command = 'git push'
    if upstream:
        command += f' --set-upstream {upstream}'
    number_of_commits = commits_to_push(self.local_dir, upstream)
    if number_of_commits > 1:
        logger.warning(f'Several commits ({number_of_commits}) will be pushed upstream.')
        if blocking:
            logger.warning('The progress bars may be unreliable.')
    try:
        with _lfs_log_progress():
            process = subprocess.Popen(command.split(), stderr=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8', cwd=self.local_dir)
            if blocking:
                stdout, stderr = process.communicate()
                return_code = process.poll()
                process.kill()
                if len(stderr):
                    logger.warning(stderr)
                if return_code:
                    raise subprocess.CalledProcessError(return_code, process.args, output=stdout, stderr=stderr)
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)
    if not blocking:

        def status_method():
            status = process.poll()
            if status is None:
                return -1
            else:
                return status
        command_in_progress = CommandInProgress('push', is_done_method=lambda: process.poll() is not None, status_method=status_method, process=process, post_method=self.lfs_prune if auto_lfs_prune else None)
        self.command_queue.append(command_in_progress)
        return (self.git_head_commit_url(), command_in_progress)
    if auto_lfs_prune:
        self.lfs_prune()
    return self.git_head_commit_url()