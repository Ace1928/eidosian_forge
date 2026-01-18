from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Issue
import github.NotificationSubject
import github.PullRequest
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def last_read_at(self) -> datetime:
    self._completeIfNotSet(self._last_read_at)
    return self._last_read_at.value