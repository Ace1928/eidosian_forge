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
def create_hook(self, name: str, config: dict[str, str], events: Opt[list[str]]=NotSet, active: Opt[bool]=NotSet) -> Hook:
    """
        :calls: `POST /orgs/{owner}/hooks <https://docs.github.com/en/rest/reference/orgs#webhooks>`_
        :param name: string
        :param config: dict
        :param events: list of string
        :param active: bool
        :rtype: :class:`github.Hook.Hook`
        """
    assert isinstance(name, str), name
    assert isinstance(config, dict), config
    assert is_optional_list(events, str), events
    assert is_optional(active, bool), active
    post_parameters: dict[str, Any] = NotSet.remove_unset_items({'name': name, 'config': config, 'events': events, 'active': active})
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/hooks', input=post_parameters)
    return github.Hook.Hook(self._requester, headers, data, completed=True)