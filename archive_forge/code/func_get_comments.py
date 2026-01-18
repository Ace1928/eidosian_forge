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
def get_comments(self, since: Opt[datetime]=NotSet) -> PaginatedList[IssueComment]:
    """
        :calls: `GET /repos/{owner}/{repo}/issues/{number}/comments <https://docs.github.com/en/rest/reference/issues#comments>`_
        """
    url_parameters = {}
    if is_defined(since):
        assert isinstance(since, datetime), since
        url_parameters['since'] = since.strftime('%Y-%m-%dT%H:%M:%SZ')
    return PaginatedList(github.IssueComment.IssueComment, self._requester, f'{self.url}/comments', url_parameters)