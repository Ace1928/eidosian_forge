from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..certificates import BoundCertificate
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..load_balancer_types import BoundLoadBalancerType
from ..locations import BoundLocation
from ..metrics import Metrics
from ..networks import BoundNetwork
from ..servers import BoundServer
from .domain import (
def enable_public_interface(self, load_balancer: LoadBalancer | BoundLoadBalancer) -> BoundAction:
    """Enables the public interface of a Load Balancer.

        :param load_balancer: :class:` <hcloud.load_balancers.client.BoundLoadBalancer>` or :class:`LoadBalancer <hcloud.load_balancers.domain.LoadBalancer>`

        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
    response = self._client.request(url=f'/load_balancers/{load_balancer.id}/actions/enable_public_interface', method='POST')
    return BoundAction(self._client.actions, response['action'])