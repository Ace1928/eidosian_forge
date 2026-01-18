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
def attach_iso(self, server: Server | BoundServer, iso: Iso | BoundIso) -> BoundAction:
    """Attaches an ISO to a server.

        :param server: :class:`BoundServer <hcloud.servers.client.BoundServer>` or :class:`Server <hcloud.servers.domain.Server>`
        :param iso: :class:`BoundIso <hcloud.isos.client.BoundIso>` or :class:`Server <hcloud.isos.domain.Iso>`
        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
    data: dict[str, Any] = {'iso': iso.id_or_name}
    response = self._client.request(url=f'/servers/{server.id}/actions/attach_iso', method='POST', json=data)
    return BoundAction(self._client.actions, response['action'])