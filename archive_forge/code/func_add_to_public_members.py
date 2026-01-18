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
def add_to_public_members(self, public_member: NamedUser) -> None:
    """
        :calls: `PUT /orgs/{org}/public_members/{user} <https://docs.github.com/en/rest/reference/orgs#members>`_
        """
    assert isinstance(public_member, github.NamedUser.NamedUser), public_member
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/public_members/{public_member._identity}')