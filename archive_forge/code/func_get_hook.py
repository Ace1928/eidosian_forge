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
def get_hook(self, id: int) -> github.Hook.Hook:
    """
        :calls: `GET /orgs/{owner}/hooks/{id} <https://docs.github.com/en/rest/reference/orgs#webhooks>`_
        """
    assert isinstance(id, int), id
    headers, data = self._requester.requestJsonAndCheck('GET', f'{self.url}/hooks/{id}')
    return github.Hook.Hook(self._requester, headers, data, completed=True)