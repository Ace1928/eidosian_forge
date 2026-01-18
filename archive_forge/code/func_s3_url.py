from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def s3_url(self) -> str:
    self._completeIfNotSet(self._s3_url)
    return self._s3_url.value