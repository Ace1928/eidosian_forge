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
def convert_to_outside_collaborator(self, member: NamedUser) -> None:
    """
        :calls: `PUT /orgs/{org}/outside_collaborators/{username} <https://docs.github.com/en/rest/reference/orgs#outside-collaborators>`_
        :param member: :class:`github.NamedUser.NamedUser`
        :rtype: None
        """
    assert isinstance(member, github.NamedUser.NamedUser), member
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/outside_collaborators/{member._identity}')