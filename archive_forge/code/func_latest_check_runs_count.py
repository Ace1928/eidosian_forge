from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.CheckRun
import github.GitCommit
import github.GithubApp
import github.PullRequest
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_defined, is_optional
from github.PaginatedList import PaginatedList
@property
def latest_check_runs_count(self) -> int:
    """
        :type: int
        """
    self._completeIfNotSet(self._latest_check_runs_count)
    return self._latest_check_runs_count.value