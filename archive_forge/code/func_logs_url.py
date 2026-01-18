from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.WorkflowStep
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
def logs_url(self) -> str:
    headers, _ = self._requester.requestBlobAndCheck('GET', f'{self.url}/logs')
    return headers['location']