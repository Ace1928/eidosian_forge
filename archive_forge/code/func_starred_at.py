from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def starred_at(self) -> datetime:
    return self._starred_at.value