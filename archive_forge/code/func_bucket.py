from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def bucket(self) -> str:
    self._completeIfNotSet(self._bucket)
    return self._bucket.value