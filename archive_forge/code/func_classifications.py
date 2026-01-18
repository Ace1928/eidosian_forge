from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.CodeScanAlertInstanceLocation
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def classifications(self) -> list[str]:
    return self._classifications.value