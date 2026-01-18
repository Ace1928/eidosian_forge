from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def permissions(self) -> dict[str, str]:
    self._completeIfNotSet(self._permissions)
    return self._permissions.value