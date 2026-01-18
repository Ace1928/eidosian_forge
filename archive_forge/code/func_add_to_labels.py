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
def add_to_labels(self, *labels: Label | str) -> None:
    """
        :calls: `POST /repos/{owner}/{repo}/issues/{number}/labels <https://docs.github.com/en/rest/reference/issues#labels>`_
        """
    assert all((isinstance(element, (github.Label.Label, str)) for element in labels)), labels
    post_parameters = [label.name if isinstance(label, github.Label.Label) else label for label in labels]
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/labels', input=post_parameters)