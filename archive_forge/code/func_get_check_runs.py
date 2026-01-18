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
def get_check_runs(self, check_name: Opt[str]=NotSet, status: Opt[str]=NotSet, filter: Opt[str]=NotSet) -> PaginatedList[CheckRun]:
    """
        :calls: `GET /repos/{owner}/{repo}/check-suites/{check_suite_id}/check-runs <https://docs.github.com/en/rest/reference/checks#list-check-runs-in-a-check-suite>`_
        """
    assert is_optional(check_name, str), check_name
    assert is_optional(status, str), status
    assert is_optional(filter, str), filter
    url_parameters: dict[str, Any] = {}
    if is_defined(check_name):
        url_parameters['check_name'] = check_name
    if is_defined(status):
        url_parameters['status'] = status
    if is_defined(filter):
        url_parameters['filter'] = filter
    return PaginatedList(github.CheckRun.CheckRun, self._requester, f'{self.url}/check-runs', url_parameters, headers={'Accept': 'application/vnd.github.v3+json'}, list_item='check_runs')