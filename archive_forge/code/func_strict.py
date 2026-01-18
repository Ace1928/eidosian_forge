from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def strict(self) -> bool:
    self._completeIfNotSet(self._strict)
    return self._strict.value