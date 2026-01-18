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
def get_public_key(self) -> PublicKey:
    """
        :calls: `GET /orgs/{org}/actions/secrets/public-key <https://docs.github.com/en/rest/reference/actions#get-an-organization-public-key>`_
        :rtype: :class:`github.PublicKey.PublicKey`
        """
    headers, data = self._requester.requestJsonAndCheck('GET', f'{self.url}/actions/secrets/public-key')
    return github.PublicKey.PublicKey(self._requester, headers, data, completed=True)