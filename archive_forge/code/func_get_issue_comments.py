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
def get_issue_comments(self) -> PaginatedList[github.IssueComment.IssueComment]:
    """
        :calls: `GET /repos/{owner}/{repo}/issues/{number}/comments <https://docs.github.com/en/rest/reference/issues#comments>`_
        """
    return PaginatedList(github.IssueComment.IssueComment, self._requester, f'{self.issue_url}/comments', None)