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
def allow_deletions(self) -> bool:
    self._completeIfNotSet(self._allow_deletions)
    return self._allow_deletions.value