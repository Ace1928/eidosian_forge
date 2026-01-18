from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def status_text(self) -> str:
    self._completeIfNotSet(self._status_text)
    return self._status_text.value