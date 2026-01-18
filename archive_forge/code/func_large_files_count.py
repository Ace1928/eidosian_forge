from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def large_files_count(self) -> int:
    self._completeIfNotSet(self._large_files_count)
    return self._large_files_count.value