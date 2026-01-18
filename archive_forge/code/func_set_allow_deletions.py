from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.BranchProtection
import github.Commit
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
from github import Consts
from github.GithubObject import (
def set_allow_deletions(self) -> None:
    """
        :calls: `POST /repos/{owner}/{repo}/branches/{branch}/protection/allow_deletions <https://docs.github.com/en/rest/reference/repos#branches>`_
        """
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.protection_url}/allow_deletions')