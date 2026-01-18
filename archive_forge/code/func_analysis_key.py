from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.CodeScanAlertInstanceLocation
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def analysis_key(self) -> str:
    return self._analysis_key.value