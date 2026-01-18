import git
from git.exc import InvalidGitRepositoryError
from git.config import GitConfigParser
from io import BytesIO
import weakref
from typing import Any, Sequence, TYPE_CHECKING, Union
from git.types import PathLike
def sm_name(section: str) -> str:
    """:return: Name of the submodule as parsed from the section name"""
    section = section.strip()
    return section[11:-1]