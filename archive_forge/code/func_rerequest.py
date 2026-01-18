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
def rerequest(self) -> bool:
    """
        :calls: `POST /repos/{owner}/{repo}/check-suites/{check_suite_id}/rerequest <https://docs.github.com/en/rest/reference/checks#rerequest-a-check-suite>`_
        :rtype: bool
        """
    request_headers = {'Accept': 'application/vnd.github.v3+json'}
    status, _, _ = self._requester.requestJson('POST', f'{self.url}/rerequest', headers=request_headers)
    return status == 201