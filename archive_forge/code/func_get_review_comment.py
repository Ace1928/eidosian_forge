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
def get_review_comment(self, id: int) -> github.PullRequestComment.PullRequestComment:
    """
        :calls: `GET /repos/{owner}/{repo}/pulls/comments/{number} <https://docs.github.com/en/rest/reference/pulls#review-comments>`_
        """
    assert isinstance(id, int), id
    headers, data = self._requester.requestJsonAndCheck('GET', f'{self._parentUrl(self.url)}/comments/{id}')
    return github.PullRequestComment.PullRequestComment(self._requester, headers, data, completed=True)