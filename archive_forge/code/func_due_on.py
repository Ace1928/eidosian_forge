from __future__ import annotations
from datetime import date, datetime
from typing import Any
import github.GithubObject
import github.Label
import github.NamedUser
import github.PaginatedList
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_defined
from github.PaginatedList import PaginatedList
@property
def due_on(self) -> datetime | None:
    self._completeIfNotSet(self._due_on)
    return self._due_on.value