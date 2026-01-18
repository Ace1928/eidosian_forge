from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def authors_count(self) -> int:
    self._completeIfNotSet(self._authors_count)
    return self._authors_count.value