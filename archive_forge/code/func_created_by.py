from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def created_by(self) -> str:
    return self._created_by.value