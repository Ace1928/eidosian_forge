from __future__ import annotations
import base64
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, _ValuedAttribute
@property
def license(self) -> License:
    self._completeIfNotSet(self._license)
    return self._license.value