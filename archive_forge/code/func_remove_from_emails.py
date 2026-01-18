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
def remove_from_emails(self, *emails: str) -> None:
    """
        :calls: `DELETE /user/emails <http://docs.github.com/en/rest/reference/users#emails>`_
        """
    assert all((isinstance(element, str) for element in emails)), emails
    post_parameters = {'emails': emails}
    headers, data = self._requester.requestJsonAndCheck('DELETE', '/user/emails', input=post_parameters)