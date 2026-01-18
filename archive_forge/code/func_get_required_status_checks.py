from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.BranchProtection
import github.Commit
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
from github import Consts
from github.GithubObject import (
def get_required_status_checks(self) -> RequiredStatusChecks:
    """
        :calls: `GET /repos/{owner}/{repo}/branches/{branch}/protection/required_status_checks <https://docs.github.com/en/rest/reference/repos#branches>`_
        :rtype: :class:`github.RequiredStatusChecks.RequiredStatusChecks`
        """
    headers, data = self._requester.requestJsonAndCheck('GET', f'{self.protection_url}/required_status_checks')
    return github.RequiredStatusChecks.RequiredStatusChecks(self._requester, headers, data, completed=True)