from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def security_severity_level(self) -> str:
    return self._security_severity_level.value