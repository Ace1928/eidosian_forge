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
def add_to_placement_group(self, server: Server | BoundServer, placement_group: PlacementGroup | BoundPlacementGroup) -> BoundAction:
    """Adds a server to a placement group.

        :param server: :class:`BoundServer <hcloud.servers.client.BoundServer>` or :class:`Server <hcloud.servers.domain.Server>`
        :param placement_group: :class:`BoundPlacementGroup <hcloud.placement_groups.client.BoundPlacementGroup>` or :class:`Network <hcloud.placement_groups.domain.PlacementGroup>`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
    data: dict[str, Any] = {'placement_group': str(placement_group.id)}
    response = self._client.request(url=f'/servers/{server.id}/actions/add_to_placement_group', method='POST', json=data)
    return BoundAction(self._client.actions, response['action'])