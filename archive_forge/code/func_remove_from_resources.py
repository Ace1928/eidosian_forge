from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import (
def remove_from_resources(self, firewall: Firewall | BoundFirewall, resources: list[FirewallResource]) -> list[BoundAction]:
    """Removes one Firewall from multiple resources.

        :param firewall: :class:`BoundFirewall <hcloud.firewalls.client.BoundFirewall>` or  :class:`Firewall <hcloud.firewalls.domain.Firewall>`
        :param resources: List[:class:`FirewallResource <hcloud.firewalls.domain.FirewallResource>`]
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
    data: dict[str, Any] = {'remove_from': []}
    for resource in resources:
        data['remove_from'].append(resource.to_payload())
    response = self._client.request(url=f'/firewalls/{firewall.id}/actions/remove_from_resources', method='POST', json=data)
    return [BoundAction(self._client.actions, action_data) for action_data in response['actions']]