from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.Commit
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def tarball_url(self) -> str:
    return self._tarball_url.value