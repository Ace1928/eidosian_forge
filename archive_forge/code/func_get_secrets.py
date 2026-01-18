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
def get_secrets(self, secret_type: str='actions') -> PaginatedList[OrganizationSecret]:
    """
        Gets all organization secrets
        :param secret_type: string options actions or dependabot
        :rtype: :class:`PaginatedList` of :class:`github.OrganizationSecret.OrganizationSecret`
        """
    assert secret_type in ['actions', 'dependabot'], 'secret_type should be actions or dependabot'
    return PaginatedList(github.OrganizationSecret.OrganizationSecret, self._requester, f'{self.url}/{secret_type}/secrets', None, list_item='secrets')