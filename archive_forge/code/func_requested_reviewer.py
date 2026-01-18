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
def requested_reviewer(self) -> github.NamedUser.NamedUser:
    self._completeIfNotSet(self._requested_reviewer)
    return self._requested_reviewer.value