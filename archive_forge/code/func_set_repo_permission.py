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
@deprecated(reason='\n        Team.set_repo_permission() is deprecated, use Team.update_team_repository() instead.\n        ')
def set_repo_permission(self, repo: Repository, permission: str) -> None:
    """
        :calls: `PUT /teams/{id}/repos/{org}/{repo} <https://docs.github.com/en/rest/reference/teams>`_
        :param repo: :class:`github.Repository.Repository`
        :param permission: string
        :rtype: None
        """
    assert isinstance(repo, github.Repository.Repository), repo
    put_parameters = {'permission': permission}
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/repos/{repo._identity}', input=put_parameters)