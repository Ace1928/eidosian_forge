from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.Event
import github.GithubObject
import github.HookDelivery
import github.NamedUser
import github.OrganizationDependabotAlert
import github.OrganizationSecret
import github.OrganizationVariable
import github.Plan
import github.Project
import github.Repository
import github.Team
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def get_public_members(self) -> PaginatedList[NamedUser]:
    """
        :calls: `GET /orgs/{org}/public_members <https://docs.github.com/en/rest/reference/orgs#members>`_
        :rtype: :class:`PaginatedList` of :class:`github.NamedUser.NamedUser`
        """
    return PaginatedList(github.NamedUser.NamedUser, self._requester, f'{self.url}/public_members', None)