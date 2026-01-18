from __future__ import annotations
from datetime import datetime
from typing import Any
import github.CommitStats
import github.Gist
import github.GithubObject
import github.NamedUser
from github.GistFile import GistFile
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def forks(self) -> list[github.Gist.Gist]:
    self._completeIfNotSet(self._forks)
    return self._forks.value