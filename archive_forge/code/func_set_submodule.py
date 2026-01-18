import git
from git.exc import InvalidGitRepositoryError
from git.config import GitConfigParser
from io import BytesIO
import weakref
from typing import Any, Sequence, TYPE_CHECKING, Union
from git.types import PathLike
def set_submodule(self, submodule: 'Submodule') -> None:
    """Set this instance's submodule. It must be called before
        the first write operation begins."""
    self._smref = weakref.ref(submodule)