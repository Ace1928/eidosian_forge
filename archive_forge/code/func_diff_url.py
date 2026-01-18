from __future__ import annotations
from typing import Any
import github.Commit
import github.File
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
@property
def diff_url(self) -> str:
    self._completeIfNotSet(self._diff_url)
    return self._diff_url.value