from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.NamedUser
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def invitee(self) -> NamedUser:
    self._completeIfNotSet(self._invitee)
    return self._invitee.value