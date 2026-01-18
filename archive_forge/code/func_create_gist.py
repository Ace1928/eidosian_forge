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
def create_gist(self, public: bool, files: dict[str, InputFileContent], description: Opt[str]=NotSet) -> Gist:
    """
        :calls: `POST /gists <http://docs.github.com/en/rest/reference/gists>`_
        """
    assert isinstance(public, bool), public
    assert all((isinstance(element, github.InputFileContent) for element in files.values())), files
    assert is_undefined(description) or isinstance(description, str), description
    post_parameters = {'public': public, 'files': {key: value._identity for key, value in files.items()}}
    if is_defined(description):
        post_parameters['description'] = description
    headers, data = self._requester.requestJsonAndCheck('POST', '/gists', input=post_parameters)
    return github.Gist.Gist(self._requester, headers, data, completed=True)