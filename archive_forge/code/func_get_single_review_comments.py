from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import NotRequired, TypedDict
import github.Commit
import github.File
import github.IssueComment
import github.IssueEvent
import github.Label
import github.Milestone
import github.NamedUser
import github.PaginatedList
import github.PullRequestComment
import github.PullRequestMergeStatus
import github.PullRequestPart
import github.PullRequestReview
import github.Team
from github import Consts
from github.GithubObject import (
from github.Issue import Issue
from github.PaginatedList import PaginatedList
def get_single_review_comments(self, id: int) -> PaginatedList[github.PullRequestComment.PullRequestComment]:
    """
        :calls: `GET /repos/{owner}/{repo}/pulls/{number}/review/{id}/comments <https://docs.github.com/en/rest/reference/pulls#reviews>`_
        """
    assert isinstance(id, int), id
    return PaginatedList(github.PullRequestComment.PullRequestComment, self._requester, f'{self.url}/reviews/{id}/comments', None)