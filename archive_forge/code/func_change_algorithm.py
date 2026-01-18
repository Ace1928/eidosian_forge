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
def change_algorithm(self, load_balancer: LoadBalancer | BoundLoadBalancer, algorithm: LoadBalancerAlgorithm) -> BoundAction:
    """Changes the algorithm used by the Load Balancer

        :param load_balancer: :class:` <hcloud.load_balancers.client.BoundLoadBalancer>` or :class:`LoadBalancer <hcloud.load_balancers.domain.LoadBalancer>`
        :param algorithm: :class:`LoadBalancerAlgorithm <hcloud.load_balancers.domain.LoadBalancerAlgorithm>`
                       The LoadBalancerSubnet you want to add to the Load Balancer
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
    data: dict[str, Any] = {'type': algorithm.type}
    response = self._client.request(url=f'/load_balancers/{load_balancer.id}/actions/change_algorithm', method='POST', json=data)
    return BoundAction(self._client.actions, response['action'])