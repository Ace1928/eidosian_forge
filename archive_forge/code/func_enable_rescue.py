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
def enable_rescue(self, server: Server | BoundServer, type: str | None=None, ssh_keys: list[str] | None=None) -> EnableRescueResponse:
    """Enable the Hetzner Rescue System for this server.

        :param server: :class:`BoundServer <hcloud.servers.client.BoundServer>` or :class:`Server <hcloud.servers.domain.Server>`
        :param type: str
                Type of rescue system to boot (default: linux64)
                Choices: linux64, linux32, freebsd64
        :param ssh_keys: List[str]
                Array of SSH key IDs which should be injected into the rescue system. Only available for types: linux64 and linux32.
        :return: :class:`EnableRescueResponse <hcloud.servers.domain.EnableRescueResponse>`
        """
    data: dict[str, Any] = {'type': type}
    if ssh_keys is not None:
        data.update({'ssh_keys': ssh_keys})
    response = self._client.request(url=f'/servers/{server.id}/actions/enable_rescue', method='POST', json=data)
    return EnableRescueResponse(action=BoundAction(self._client.actions, response['action']), root_password=response['root_password'])