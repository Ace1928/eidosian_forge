from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def comments_count(self) -> int:
    self._completeIfNotSet(self._comments_count)
    return self._comments_count.value