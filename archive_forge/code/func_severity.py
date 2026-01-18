from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def severity(self) -> str:
    return self._severity.value