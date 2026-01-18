from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def submitted_at(self) -> datetime:
    return self._submitted_at.value