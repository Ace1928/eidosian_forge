from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def supported_events(self) -> list[str]:
    return self._supported_events.value