from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
import github.GitCommit
import github.PullRequest
import github.WorkflowJob
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
from github.PaginatedList import PaginatedList
def rerun(self) -> bool:
    """
        :calls: `POST /repos/{owner}/{repo}/actions/runs/{run_id}/rerun <https://docs.github.com/en/rest/reference/actions#workflow-runs>`_
        """
    status, _, _ = self._requester.requestJson('POST', self.rerun_url)
    return status == 201