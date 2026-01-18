from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.IssueComment
import github.IssueEvent
import github.IssuePullRequest
import github.Label
import github.Milestone
import github.NamedUser
import github.PullRequest
import github.Reaction
import github.Repository
import github.TimelineEvent
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
@property
def state_reason(self) -> str | None:
    self._completeIfNotSet(self._state_reason)
    return self._state_reason.value