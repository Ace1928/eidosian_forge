from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.WorkflowStep
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def run_url(self) -> str:
    self._completeIfNotSet(self._run_url)
    return self._run_url.value