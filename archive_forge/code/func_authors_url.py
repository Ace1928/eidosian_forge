from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def authors_url(self) -> str:
    self._completeIfNotSet(self._authors_url)
    return self._authors_url.value