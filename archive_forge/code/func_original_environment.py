from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Consts
import github.DeploymentStatus
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
@property
def original_environment(self) -> str:
    self._completeIfNotSet(self._original_environment)
    return self._original_environment.value