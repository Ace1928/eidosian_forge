from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.BranchProtection
import github.Commit
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
from github import Consts
from github.GithubObject import (
def edit_required_status_checks(self, strict: Opt[bool]=NotSet, contexts: Opt[list[str]]=NotSet) -> RequiredStatusChecks:
    """
        :calls: `PATCH /repos/{owner}/{repo}/branches/{branch}/protection/required_status_checks <https://docs.github.com/en/rest/reference/repos#branches>`_
        """
    assert is_optional(strict, bool), strict
    assert is_optional_list(contexts, str), contexts
    post_parameters: dict[str, Any] = NotSet.remove_unset_items({'strict': strict, 'contexts': contexts})
    headers, data = self._requester.requestJsonAndCheck('PATCH', f'{self.protection_url}/required_status_checks', input=post_parameters)
    return github.RequiredStatusChecks.RequiredStatusChecks(self._requester, headers, data, completed=True)