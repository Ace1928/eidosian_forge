import git
from git.exc import InvalidGitRepositoryError
from git.config import GitConfigParser
from io import BytesIO
import weakref
from typing import Any, Sequence, TYPE_CHECKING, Union
from git.types import PathLike
def find_first_remote_branch(remotes: Sequence['Remote'], branch_name: str) -> 'RemoteReference':
    """Find the remote branch matching the name of the given branch or raise InvalidGitRepositoryError."""
    for remote in remotes:
        try:
            return remote.refs[branch_name]
        except IndexError:
            continue
    raise InvalidGitRepositoryError("Didn't find remote branch '%r' in any of the given remotes" % branch_name)