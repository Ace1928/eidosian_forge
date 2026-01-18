from git.config import GitConfigParser, SectionConstraint
from git.util import join_path
from git.exc import GitCommandError
from .symbolic import SymbolicReference
from .reference import Reference
from typing import Any, Sequence, Union, TYPE_CHECKING
from git.types import PathLike, Commit_ish
def orig_head(self) -> SymbolicReference:
    """
        :return: SymbolicReference pointing at the ORIG_HEAD, which is maintained
            to contain the previous value of HEAD.
        """
    return SymbolicReference(self.repo, self._ORIG_HEAD_NAME)