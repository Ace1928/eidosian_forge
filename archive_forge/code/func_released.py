from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def released(self) -> str:
    return self._released.value