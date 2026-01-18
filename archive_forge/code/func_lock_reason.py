from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Issue
import github.Label
import github.Milestone
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def lock_reason(self) -> str:
    self._completeIfNotSet(self._lock_reason)
    return self._lock_reason.value