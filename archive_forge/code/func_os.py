from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def os(self) -> str:
    return self._os.value