from __future__ import annotations
from typing import Any
import github.Commit
import github.File
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
@property
def base_commit(self) -> github.Commit.Commit:
    self._completeIfNotSet(self._base_commit)
    return self._base_commit.value