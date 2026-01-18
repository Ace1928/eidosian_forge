from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.PaginatedList
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def on_behalf_of(self) -> NamedUser:
    return self._on_behalf_of.value