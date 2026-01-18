from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.CheckRun
import github.CheckSuite
import github.CommitCombinedStatus
import github.CommitComment
import github.CommitStats
import github.CommitStatus
import github.File
import github.GitCommit
import github.NamedUser
import github.PaginatedList
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
from github.PaginatedList import PaginatedList
def get_check_suites(self, app_id: Opt[int]=NotSet, check_name: Opt[str]=NotSet) -> PaginatedList[CheckSuite]:
    """
        :class: `GET /repos/{owner}/{repo}/commits/{ref}/check-suites <https://docs.github.com/en/rest/reference/checks#list-check-suites-for-a-git-reference>`_
        """
    assert is_optional(app_id, int), app_id
    assert is_optional(check_name, str), check_name
    parameters = NotSet.remove_unset_items({'app_id': app_id, 'check_name': check_name})
    request_headers = {'Accept': 'application/vnd.github.v3+json'}
    return PaginatedList(github.CheckSuite.CheckSuite, self._requester, f'{self.url}/check-suites', parameters, headers=request_headers, list_item='check_suites')