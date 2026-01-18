from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.WorkflowStep
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def completed_at(self) -> datetime:
    self._completeIfNotSet(self._completed_at)
    return self._completed_at.value