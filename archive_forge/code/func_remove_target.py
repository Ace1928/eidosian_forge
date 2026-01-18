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
def remove_target(self, load_balancer: LoadBalancer | BoundLoadBalancer, target: LoadBalancerTarget) -> BoundAction:
    """Removes a target from a Load Balancer.

        :param load_balancer: :class:`BoundLoadBalancer <hcloud.load_balancers.client.BoundLoadBalancer>` or :class:`LoadBalancer <hcloud.load_balancers.domain.LoadBalancer>`
        :param target: :class:`LoadBalancerTarget <hcloud.load_balancers.domain.LoadBalancerTarget>`
                       The LoadBalancerTarget you want to remove from the Load Balancer
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
    data: dict[str, Any] = target.to_payload()
    data.pop('use_private_ip', None)
    response = self._client.request(url=f'/load_balancers/{load_balancer.id}/actions/remove_target', method='POST', json=data)
    return BoundAction(self._client.actions, response['action'])