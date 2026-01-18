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
def remove_from_labels(self, label: Label | str) -> None:
    """
        :calls: `DELETE /repos/{owner}/{repo}/issues/{number}/labels/{name} <https://docs.github.com/en/rest/reference/issues#labels>`_
        """
    assert isinstance(label, (github.Label.Label, str)), label
    if isinstance(label, github.Label.Label):
        label = label._identity
    else:
        label = urllib.parse.quote(label)
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/labels/{label}')