from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.Authorization
import github.Event
import github.Gist
import github.GithubObject
import github.Issue
import github.Notification
import github.Organization
import github.PaginatedList
import github.Plan
import github.Repository
import github.UserKey
from github import Consts
from github.Auth import AppAuth
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
from github.Requester import Requester
@property
def target_id(self) -> int:
    return self._target_id.value