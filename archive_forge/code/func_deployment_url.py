from __future__ import annotations
from datetime import datetime
from typing import Any
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def deployment_url(self) -> str:
    self._completeIfNotSet(self._deployment_url)
    return self._deployment_url.value