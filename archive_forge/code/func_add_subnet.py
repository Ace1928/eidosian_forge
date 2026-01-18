from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Network, NetworkRoute, NetworkSubnet
def add_subnet(self, network: Network | BoundNetwork, subnet: NetworkSubnet) -> BoundAction:
    """Adds a subnet entry to a network.

        :param network: :class:`BoundNetwork <hcloud.networks.client.BoundNetwork>` or :class:`Network <hcloud.networks.domain.Network>`
        :param subnet: :class:`NetworkSubnet <hcloud.networks.domain.NetworkSubnet>`
                       The NetworkSubnet you want to add to the Network
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
    data: dict[str, Any] = {'type': subnet.type, 'network_zone': subnet.network_zone}
    if subnet.ip_range is not None:
        data['ip_range'] = subnet.ip_range
    if subnet.vswitch_id is not None:
        data['vswitch_id'] = subnet.vswitch_id
    response = self._client.request(url=f'/networks/{network.id}/actions/add_subnet', method='POST', json=data)
    return BoundAction(self._client.actions, response['action'])