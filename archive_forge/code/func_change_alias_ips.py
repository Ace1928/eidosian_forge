from __future__ import annotations
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..datacenters import BoundDatacenter
from ..firewalls import BoundFirewall
from ..floating_ips import BoundFloatingIP
from ..images import BoundImage, CreateImageResponse
from ..isos import BoundIso
from ..metrics import Metrics
from ..placement_groups import BoundPlacementGroup
from ..primary_ips import BoundPrimaryIP
from ..server_types import BoundServerType
from ..volumes import BoundVolume
from .domain import (
def change_alias_ips(self, server: Server | BoundServer, network: Network | BoundNetwork, alias_ips: list[str]) -> BoundAction:
    """Changes the alias IPs of an already attached network.

        :param server: :class:`BoundServer <hcloud.servers.client.BoundServer>` or :class:`Server <hcloud.servers.domain.Server>`
        :param network: :class:`BoundNetwork <hcloud.networks.client.BoundNetwork>` or :class:`Network <hcloud.networks.domain.Network>`
        :param alias_ips: List[str]
                New alias IPs to set for this server.
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
    data: dict[str, Any] = {'network': network.id, 'alias_ips': alias_ips}
    response = self._client.request(url=f'/servers/{server.id}/actions/change_alias_ips', method='POST', json=data)
    return BoundAction(self._client.actions, response['action'])