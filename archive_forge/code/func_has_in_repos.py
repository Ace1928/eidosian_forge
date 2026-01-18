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
def has_in_repos(self, repo: Repository) -> bool:
    """
        :calls: `GET /teams/{id}/repos/{owner}/{repo} <https://docs.github.com/en/rest/reference/teams>`_
        """
    assert isinstance(repo, github.Repository.Repository), repo
    status, headers, data = self._requester.requestJson('GET', f'{self.url}/repos/{repo._identity}')
    return status == 204