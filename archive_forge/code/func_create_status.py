from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Consts
import github.DeploymentStatus
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
def create_status(self, state: str, target_url: Opt[str]=NotSet, description: Opt[str]=NotSet, environment: Opt[str]=NotSet, environment_url: Opt[str]=NotSet, auto_inactive: Opt[bool]=NotSet) -> github.DeploymentStatus.DeploymentStatus:
    """
        :calls: `POST /repos/{owner}/{repo}/deployments/{deployment_id}/statuses <https://docs.github.com/en/rest/reference/repos#create-a-deployment-status>`_
        """
    assert isinstance(state, str), state
    assert target_url is NotSet or isinstance(target_url, str), target_url
    assert description is NotSet or isinstance(description, str), description
    assert environment is NotSet or isinstance(environment, str), environment
    assert environment_url is NotSet or isinstance(environment_url, str), environment_url
    assert auto_inactive is NotSet or isinstance(auto_inactive, bool), auto_inactive
    post_parameters = NotSet.remove_unset_items({'state': state, 'target_url': target_url, 'description': description, 'environment': environment, 'environment_url': environment_url, 'auto_inactive': auto_inactive})
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/statuses', input=post_parameters, headers={'Accept': self._get_accept_header()})
    return github.DeploymentStatus.DeploymentStatus(self._requester, headers, data, completed=True)