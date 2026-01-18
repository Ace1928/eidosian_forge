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
def add_to_watched(self, watched: Repository) -> None:
    """
        :calls: `PUT /repos/{owner}/{repo}/subscription <http://docs.github.com/en/rest/reference/activity#watching>`_
        """
    assert isinstance(watched, github.Repository.Repository), watched
    headers, data = self._requester.requestJsonAndCheck('PUT', f'/repos/{watched._identity}/subscription', input={'subscribed': True})