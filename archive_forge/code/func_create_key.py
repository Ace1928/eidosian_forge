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
def create_key(self, title: str, key: str) -> UserKey:
    """
        :calls: `POST /user/keys <http://docs.github.com/en/rest/reference/users#git-ssh-keys>`_
        :param title: string
        :param key: string
        :rtype: :class:`github.UserKey.UserKey`
        """
    assert isinstance(title, str), title
    assert isinstance(key, str), key
    post_parameters = {'title': title, 'key': key}
    headers, data = self._requester.requestJsonAndCheck('POST', '/user/keys', input=post_parameters)
    return github.UserKey.UserKey(self._requester, headers, data, completed=True)