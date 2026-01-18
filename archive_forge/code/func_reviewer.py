from __future__ import annotations
from typing import Any
import github.NamedUser
import github.Team
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def reviewer(self) -> github.NamedUser.NamedUser | github.Team.Team:
    return self._reviewer.value