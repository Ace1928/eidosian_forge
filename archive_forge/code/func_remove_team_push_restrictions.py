from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.BranchProtection
import github.Commit
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
from github import Consts
from github.GithubObject import (
def remove_team_push_restrictions(self, *teams: str) -> None:
    """
        :calls: `DELETE /repos/{owner}/{repo}/branches/{branch}/protection/restrictions/teams <https://docs.github.com/en/rest/reference/repos#branches>`_
        :teams: list of strings (team slugs)
        """
    assert all((isinstance(element, str) for element in teams)), teams
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.protection_url}/restrictions/teams', input=teams)