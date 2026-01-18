from __future__ import annotations
from typing import Any
import github.CommitStatus
import github.Repository
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def statuses(self) -> list[github.CommitStatus.CommitStatus]:
    return self._statuses.value