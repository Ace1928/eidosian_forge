from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.CodeScanAlertInstanceLocation
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def commit_sha(self) -> str:
    return self._commit_sha.value