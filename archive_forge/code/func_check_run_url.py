from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.WorkflowStep
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def check_run_url(self) -> str:
    self._completeIfNotSet(self._check_run_url)
    return self._check_run_url.value