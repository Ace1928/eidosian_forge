from __future__ import annotations
import urllib.parse
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, NamedTuple
import github.Authorization
import github.Event
import github.Gist
import github.GithubObject
import github.Invitation
import github.Issue
import github.Membership
import github.Migration
import github.NamedUser
import github.Notification
import github.Organization
import github.Plan
import github.Repository
import github.UserKey
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def has_in_watched(self, watched: Repository) -> bool:
    """
        :calls: `GET /repos/{owner}/{repo}/subscription <http://docs.github.com/en/rest/reference/activity#watching>`_
        """
    assert isinstance(watched, github.Repository.Repository), watched
    status, headers, data = self._requester.requestJson('GET', f'/repos/{watched._identity}/subscription')
    return status == 200