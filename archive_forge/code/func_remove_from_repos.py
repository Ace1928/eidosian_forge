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
def remove_from_repos(self, repo: Repository) -> None:
    """
        :calls: `DELETE /teams/{id}/repos/{owner}/{repo} <https://docs.github.com/en/rest/reference/teams>`_
        """
    assert isinstance(repo, github.Repository.Repository), repo
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/repos/{repo._identity}')