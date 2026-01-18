from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def conditions(self) -> list[str]:
    self._completeIfNotSet(self._conditions)
    return self._conditions.value