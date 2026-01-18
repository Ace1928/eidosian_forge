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
def get_review_comments(self, *, sort: Opt[str]=NotSet, direction: Opt[str]=NotSet, since: Opt[datetime]=NotSet) -> PaginatedList[github.PullRequestComment.PullRequestComment]:
    """
        :calls: `GET /repos/{owner}/{repo}/pulls/{number}/comments <https://docs.github.com/en/rest/reference/pulls#review-comments>`_
        :param sort: string 'created' or 'updated'
        :param direction: string 'asc' or 'desc'
        :param since: datetime
        """
    assert is_optional(sort, str), sort
    assert is_optional(direction, str), direction
    assert is_optional(since, datetime), since
    url_parameters = NotSet.remove_unset_items({'sort': sort, 'direction': direction})
    if is_defined(since):
        url_parameters['since'] = since.strftime('%Y-%m-%dT%H:%M:%SZ')
    return PaginatedList(github.PullRequestComment.PullRequestComment, self._requester, f'{self.url}/comments', url_parameters)