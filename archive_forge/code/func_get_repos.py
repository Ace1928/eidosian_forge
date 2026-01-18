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
def get_repos(self) -> PaginatedList[Repository]:
    """
        :calls: `GET /teams/{id}/repos <https://docs.github.com/en/rest/reference/teams>`_
        """
    return github.PaginatedList.PaginatedList(github.Repository.Repository, self._requester, f'{self.url}/repos', None)