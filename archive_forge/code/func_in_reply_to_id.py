from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
import github.Reaction
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
@property
def in_reply_to_id(self) -> int:
    self._completeIfNotSet(self._in_reply_to_id)
    return self._in_reply_to_id.value