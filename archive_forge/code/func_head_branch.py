from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
import github.GitCommit
import github.PullRequest
import github.WorkflowJob
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
from github.PaginatedList import PaginatedList
@property
def head_branch(self) -> str:
    self._completeIfNotSet(self._head_branch)
    return self._head_branch.value