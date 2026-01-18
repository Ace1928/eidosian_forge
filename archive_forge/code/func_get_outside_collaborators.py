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
def get_outside_collaborators(self, filter_: Opt[str]=NotSet) -> PaginatedList[NamedUser]:
    """
        :calls: `GET /orgs/{org}/outside_collaborators <https://docs.github.com/en/rest/reference/orgs#outside-collaborators>`_
        """
    assert is_optional(filter_, str), filter_
    url_parameters = NotSet.remove_unset_items({'filter': filter_})
    return PaginatedList(github.NamedUser.NamedUser, self._requester, f'{self.url}/outside_collaborators', url_parameters)