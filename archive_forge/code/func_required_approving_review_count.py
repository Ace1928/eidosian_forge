from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.Team
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def required_approving_review_count(self) -> int:
    self._completeIfNotSet(self._required_approving_review_count)
    return self._required_approving_review_count.value