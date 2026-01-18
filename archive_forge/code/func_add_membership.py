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
def add_membership(self, member: NamedUser, role: Opt[str]=NotSet) -> None:
    """
        :calls: `PUT /teams/{id}/memberships/{user} <https://docs.github.com/en/rest/reference/teams>`_
        """
    assert isinstance(member, github.NamedUser.NamedUser), member
    assert role is NotSet or isinstance(role, str), role
    if role is not NotSet:
        assert role in ['member', 'maintainer']
        put_parameters = {'role': role}
    else:
        put_parameters = {'role': 'member'}
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/memberships/{member._identity}', input=put_parameters)