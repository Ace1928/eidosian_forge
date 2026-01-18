from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.CheckRunAnnotation
import github.CheckRunOutput
import github.GithubApp
import github.GithubObject
import github.PullRequest
from github.GithubObject import (
from github.PaginatedList import PaginatedList
@property
def check_suite_id(self) -> int:
    self._completeIfNotSet(self._check_suite_id)
    return self._check_suite_id.value