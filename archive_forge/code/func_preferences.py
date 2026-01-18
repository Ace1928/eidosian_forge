from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.Repository
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def preferences(self) -> dict[str, list[dict[str, bool | int]]]:
    return self._preferences.value