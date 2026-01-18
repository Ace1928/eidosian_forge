from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Issue
import github.NotificationSubject
import github.PullRequest
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
def mark_as_read(self) -> None:
    """
        :calls: `PATCH /notifications/threads/{id} <https://docs.github.com/en/rest/reference/activity#notifications>`_
        """
    headers, data = self._requester.requestJsonAndCheck('PATCH', self.url)