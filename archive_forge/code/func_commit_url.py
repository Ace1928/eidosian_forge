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
def commit_url(self) -> str:
    self._completeIfNotSet(self._commit_url)
    return self._commit_url.value