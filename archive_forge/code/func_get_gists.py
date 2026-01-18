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
def get_gists(self, since: Opt[datetime]=NotSet) -> PaginatedList[Gist]:
    """
        :calls: `GET /gists <http://docs.github.com/en/rest/reference/gists>`_
        :param since: datetime format YYYY-MM-DDTHH:MM:SSZ
        :rtype: :class:`PaginatedList` of :class:`github.Gist.Gist`
        """
    assert is_optional(since, datetime), since
    url_parameters: dict[str, Any] = {}
    if is_defined(since):
        url_parameters['since'] = since.strftime('%Y-%m-%dT%H:%M:%SZ')
    return PaginatedList(github.Gist.Gist, self._requester, '/gists', url_parameters)