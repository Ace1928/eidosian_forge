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
def get_members(self, role: Opt[str]=NotSet) -> PaginatedList[NamedUser]:
    """
        :calls: `GET /teams/{id}/members <https://docs.github.com/en/rest/reference/teams#list-team-members>`_
        """
    assert role is NotSet or isinstance(role, str), role
    url_parameters: dict[str, Any] = {}
    if role is not NotSet:
        assert role in ['member', 'maintainer', 'all']
        url_parameters['role'] = role
    return github.PaginatedList.PaginatedList(github.NamedUser.NamedUser, self._requester, f'{self.url}/members', url_parameters)