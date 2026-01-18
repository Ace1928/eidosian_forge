import contextlib
import datetime
import glob
from io import BytesIO
import os
from stat import S_ISLNK
import subprocess
import tempfile
from git.compat import (
from git.exc import GitCommandError, CheckoutError, GitError, InvalidGitRepositoryError
from git.objects import (
from git.objects.util import Serializable
from git.util import (
from gitdb.base import IStream
from gitdb.db import MemoryDB
import git.diff as git_diff
import os.path as osp
from .fun import (
from .typ import (
from .util import TemporaryFileSwap, post_clear_cache, default_index, git_working_dir
from typing import (
from git.types import Commit_ish, PathLike
def resolve_blobs(self, iter_blobs: Iterator[Blob]) -> 'IndexFile':
    """Resolve the blobs given in blob iterator.

        This will effectively remove the index entries of the respective path at all
        non-null stages and add the given blob as new stage null blob.

        For each path there may only be one blob, otherwise a ValueError will be raised
        claiming the path is already at stage 0.

        :raise ValueError: if one of the blobs already existed at stage 0

        :return: self

        :note:
            You will have to write the index manually once you are done, i.e.
            ``index.resolve_blobs(blobs).write()``.
        """
    for blob in iter_blobs:
        stage_null_key = (blob.path, 0)
        if stage_null_key in self.entries:
            raise ValueError('Path %r already exists at stage 0' % str(blob.path))
        for stage in (1, 2, 3):
            try:
                del self.entries[blob.path, stage]
            except KeyError:
                pass
        self.entries[stage_null_key] = IndexEntry.from_blob(blob)
    return self