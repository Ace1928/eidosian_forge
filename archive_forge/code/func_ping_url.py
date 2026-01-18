from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.HookResponse
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional, is_optional_list
@property
def ping_url(self) -> str:
    self._completeIfNotSet(self._ping_url)
    return self._ping_url.value