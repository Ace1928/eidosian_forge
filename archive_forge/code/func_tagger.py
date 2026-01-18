from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.GitAuthor
import github.GithubObject
import github.GitObject
import github.GitTreeElement
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def tagger(self) -> GitAuthor:
    self._completeIfNotSet(self._tagger)
    return self._tagger.value