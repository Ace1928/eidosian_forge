from __future__ import annotations
from datetime import datetime
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def short_description(self) -> str:
    return self._short_description.value