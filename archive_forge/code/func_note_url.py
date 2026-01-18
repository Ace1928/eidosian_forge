from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.AuthorizationApplication
import github.GithubObject
from github.GithubObject import Attribute, NotSet, Opt, _NotSetType
@property
def note_url(self) -> str | None:
    self._completeIfNotSet(self._note_url)
    return self._note_url.value