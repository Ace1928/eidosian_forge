from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def pull_request_url(self) -> str:
    return self._pull_request_url.value