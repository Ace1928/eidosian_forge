from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.AuthorizationApplication
import github.GithubObject
from github.GithubObject import Attribute, NotSet, Opt, _NotSetType
@property
def scopes(self) -> str:
    self._completeIfNotSet(self._scopes)
    return self._scopes.value