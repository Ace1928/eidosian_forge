from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def expirationdate(self) -> datetime:
    self._completeIfNotSet(self._expirationdate)
    return self._expirationdate.value