from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.Team
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def dismiss_stale_reviews(self) -> bool:
    self._completeIfNotSet(self._dismiss_stale_reviews)
    return self._dismiss_stale_reviews.value