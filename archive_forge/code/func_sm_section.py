import git
from git.exc import InvalidGitRepositoryError
from git.config import GitConfigParser
from io import BytesIO
import weakref
from typing import Any, Sequence, TYPE_CHECKING, Union
from git.types import PathLike
def sm_section(name: str) -> str:
    """:return: Section title used in .gitmodules configuration file"""
    return f'submodule "{name}"'