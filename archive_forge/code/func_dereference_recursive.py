import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
@classmethod
def dereference_recursive(cls, repo: 'Repo', ref_path: Union[PathLike, None]) -> str:
    """
        :return: hexsha stored in the reference at the given ref_path, recursively dereferencing all
            intermediate references as required

        :param repo: The repository containing the reference at ref_path
        """
    while True:
        hexsha, ref_path = cls._get_ref_info(repo, ref_path)
        if hexsha is not None:
            return hexsha