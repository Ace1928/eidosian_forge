from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def large_files_size(self) -> int:
    self._completeIfNotSet(self._large_files_size)
    return self._large_files_size.value