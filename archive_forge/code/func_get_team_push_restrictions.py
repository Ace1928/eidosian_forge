from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.BranchProtection
import github.Commit
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
from github import Consts
from github.GithubObject import (
def get_team_push_restrictions(self) -> PaginatedList[Team]:
    """
        :calls: `GET /repos/{owner}/{repo}/branches/{branch}/protection/restrictions/teams <https://docs.github.com/en/rest/reference/repos#branches>`_
        """
    return github.PaginatedList.PaginatedList(github.Team.Team, self._requester, f'{self.protection_url}/restrictions/teams', None)