from __future__ import annotations
import base64
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, _ValuedAttribute
@property
def git_url(self) -> str:
    self._completeIfNotSet(self._git_url)
    return self._git_url.value