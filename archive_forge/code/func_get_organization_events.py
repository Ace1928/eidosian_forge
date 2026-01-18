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
def get_organization_events(self, org: Organization) -> PaginatedList[Event]:
    """
        :calls: `GET /users/{user}/events/orgs/{org} <http://docs.github.com/en/rest/reference/activity#events>`_
        """
    assert isinstance(org, github.Organization.Organization), org
    return PaginatedList(github.Event.Event, self._requester, f'/users/{self.login}/events/orgs/{org.login}', None)