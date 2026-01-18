from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.WorkflowStep
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def head_sha(self) -> str:
    self._completeIfNotSet(self._head_sha)
    return self._head_sha.value