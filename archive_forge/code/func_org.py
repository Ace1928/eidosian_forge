from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
import github.Organization
import github.Repository
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def org(self) -> github.Organization.Organization:
    return self._org.value