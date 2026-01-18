from __future__ import annotations
from datetime import datetime
from typing import Any
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def environment_url(self) -> str:
    self._completeIfNotSet(self._environment_url)
    return self._environment_url.value