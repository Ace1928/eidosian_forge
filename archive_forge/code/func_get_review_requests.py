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
def get_review_requests(self) -> tuple[PaginatedList[NamedUser], PaginatedList[github.Team.Team]]:
    """
        :calls: `GET /repos/{owner}/{repo}/pulls/{number}/requested_reviewers <https://docs.github.com/en/rest/reference/pulls#review-requests>`_
        :rtype: tuple of :class:`github.PaginatedList.PaginatedList` of :class:`github.NamedUser.NamedUser` and of :class:`github.PaginatedList.PaginatedList` of :class:`github.Team.Team`
        """
    return (PaginatedList(github.NamedUser.NamedUser, self._requester, f'{self.url}/requested_reviewers', None, list_item='users'), PaginatedList(github.Team.Team, self._requester, f'{self.url}/requested_reviewers', None, list_item='teams'))