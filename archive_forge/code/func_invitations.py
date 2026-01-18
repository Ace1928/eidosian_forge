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
def invitations(self) -> PaginatedList[NamedUser]:
    """
        :calls: `GET /teams/{id}/invitations <https://docs.github.com/en/rest/reference/teams#members>`_
        """
    return github.PaginatedList.PaginatedList(github.NamedUser.NamedUser, self._requester, f'{self.url}/invitations', None, headers={'Accept': Consts.mediaTypeOrganizationInvitationPreview})