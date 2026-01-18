from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
from deprecated import deprecated
import github.NamedUser
import github.Organization
import github.PaginatedList
import github.Repository
import github.TeamDiscussion
from github import Consts
from github.GithubException import UnknownObjectException
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
def add_to_members(self, member: NamedUser) -> None:
    """
        This API call is deprecated. Use `add_membership` instead.
        https://docs.github.com/en/rest/reference/teams#add-or-update-team-membership-for-a-user-legacy

        :calls: `PUT /teams/{id}/members/{user} <https://docs.github.com/en/rest/reference/teams>`_
        """
    assert isinstance(member, github.NamedUser.NamedUser), member
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/members/{member._identity}')