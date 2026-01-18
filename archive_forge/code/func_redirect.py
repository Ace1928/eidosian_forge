from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def redirect(self) -> bool:
    self._completeIfNotSet(self._redirect)
    return self._redirect.value