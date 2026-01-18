from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def has_large_files(self) -> bool:
    self._completeIfNotSet(self._has_large_files)
    return self._has_large_files.value