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
@classmethod
def from_tree(cls, repo: 'Repo', *treeish: Treeish, **kwargs: Any) -> 'IndexFile':
    """Merge the given treeish revisions into a new index which is returned.
        The original index will remain unaltered.

        :param repo:
            The repository treeish are located in.

        :param treeish:
            One, two or three Tree Objects, Commits or 40 byte hexshas. The result
            changes according to the amount of trees.
            If 1 Tree is given, it will just be read into a new index
            If 2 Trees are given, they will be merged into a new index using a
            two way merge algorithm. Tree 1 is the 'current' tree, tree 2 is the 'other'
            one. It behaves like a fast-forward.
            If 3 Trees are given, a 3-way merge will be performed with the first tree
            being the common ancestor of tree 2 and tree 3. Tree 2 is the 'current' tree,
            tree 3 is the 'other' one.

        :param kwargs:
            Additional arguments passed to git-read-tree.

        :return:
            New IndexFile instance. It will point to a temporary index location which
            does not exist anymore. If you intend to write such a merged Index, supply
            an alternate file_path to its 'write' method.

        :note:
            In the three-way merge case, --aggressive will be specified to automatically
            resolve more cases in a commonly correct manner. Specify trivial=True as kwarg
            to override that.

            As the underlying git-read-tree command takes into account the current
            index, it will be temporarily moved out of the way to prevent any unexpected
            interference.
        """
    if len(treeish) == 0 or len(treeish) > 3:
        raise ValueError('Please specify between 1 and 3 treeish, got %i' % len(treeish))
    arg_list: List[Union[Treeish, str]] = []
    if len(treeish) > 1:
        arg_list.append('--reset')
        arg_list.append('--aggressive')
    with _named_temporary_file_for_subprocess(repo.git_dir) as tmp_index:
        arg_list.append('--index-output=%s' % tmp_index)
        arg_list.extend(treeish)
        with TemporaryFileSwap(join_path_native(repo.git_dir, 'index')):
            repo.git.read_tree(*arg_list, **kwargs)
            index = cls(repo, tmp_index)
            index.entries
            return index