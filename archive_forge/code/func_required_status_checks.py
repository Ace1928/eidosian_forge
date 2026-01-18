from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.NamedUser
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
import github.Team
from github.GithubObject import Attribute, NotSet, Opt, is_defined
from github.PaginatedList import PaginatedList
@property
def required_status_checks(self) -> RequiredStatusChecks:
    self._completeIfNotSet(self._required_status_checks)
    return self._required_status_checks.value