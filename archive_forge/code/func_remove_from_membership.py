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
def remove_from_membership(self, member: NamedUser) -> None:
    """
        :calls: `DELETE /orgs/{org}/memberships/{user} <https://docs.github.com/en/rest/reference/orgs#remove-an-organization-member>`_
        :param member: :class:`github.NamedUser.NamedUser`
        :rtype: None
        """
    assert isinstance(member, github.NamedUser.NamedUser), member
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/memberships/{member._identity}')