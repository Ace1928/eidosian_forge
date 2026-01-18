from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Issue
import github.NamedUser
import github.ProjectColumn
import github.PullRequest
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
@property
def archived(self) -> bool:
    return self._archived.value