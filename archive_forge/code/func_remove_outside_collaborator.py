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
def remove_outside_collaborator(self, collaborator: NamedUser) -> None:
    """
        :calls: `DELETE /orgs/{org}/outside_collaborators/{username} <https://docs.github.com/en/rest/reference/orgs#outside-collaborators>`_
        :param collaborator: :class:`github.NamedUser.NamedUser`
        :rtype: None
        """
    assert isinstance(collaborator, github.NamedUser.NamedUser), collaborator
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/outside_collaborators/{collaborator._identity}')