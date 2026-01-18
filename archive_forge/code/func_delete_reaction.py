from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.IssueComment
import github.IssueEvent
import github.IssuePullRequest
import github.Label
import github.Milestone
import github.NamedUser
import github.PullRequest
import github.Reaction
import github.Repository
import github.TimelineEvent
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def delete_reaction(self, reaction_id: int) -> bool:
    """
        :calls: `DELETE /repos/{owner}/{repo}/issues/{issue_number}/reactions/{reaction_id} <https://docs.github.com/en/rest/reference/reactions#delete-an-issue-reaction>`_
        """
    assert isinstance(reaction_id, int), reaction_id
    status, _, _ = self._requester.requestJson('DELETE', f'{self.url}/reactions/{reaction_id}', headers={'Accept': Consts.mediaTypeReactionsPreview})
    return status == 204