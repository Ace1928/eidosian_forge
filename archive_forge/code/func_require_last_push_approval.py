from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.Team
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def require_last_push_approval(self) -> bool:
    self._completeIfNotSet(self._require_last_push_approval)
    return self._require_last_push_approval.value