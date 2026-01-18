from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.Team
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def dismissal_teams(self) -> list[Team]:
    self._completeIfNotSet(self._teams)
    return self._teams.value