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
def create_authorization(self, scopes: Opt[list[str]]=NotSet, note: Opt[str]=NotSet, note_url: Opt[str]=NotSet, client_id: Opt[str]=NotSet, client_secret: Opt[str]=NotSet, onetime_password: str | None=None) -> Authorization:
    """
        :calls: `POST /authorizations <https://docs.github.com/en/developers/apps/authorizing-oauth-apps>`_
        """
    assert is_optional_list(scopes, str), scopes
    assert is_optional(note, str), note
    assert is_optional(note_url, str), note_url
    assert is_optional(client_id, str), client_id
    assert is_optional(client_secret, str), client_secret
    assert onetime_password is None or isinstance(onetime_password, str), onetime_password
    post_parameters: dict[str, Any] = NotSet.remove_unset_items({'scopes': scopes, 'note': note, 'note_url': note_url, 'client_id': client_id, 'client_secret': client_secret})
    if onetime_password is not None:
        request_header = {Consts.headerOTP: onetime_password}
    else:
        request_header = None
    headers, data = self._requester.requestJsonAndCheck('POST', '/authorizations', input=post_parameters, headers=request_header)
    return github.Authorization.Authorization(self._requester, headers, data, completed=True)