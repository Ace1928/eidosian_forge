from collections import defaultdict, deque
import logging
import platform
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
import ray
from ray.actor import ActorClass, ActorHandle
def split_colocated(actors: List[ActorHandle], node: Optional[str]='localhost') -> Tuple[List[ActorHandle], List[ActorHandle]]:
    """Splits up given actors into colocated (on same node) and non colocated.

    The co-location criterion depends on the `node` given:
    If given (or default: platform.node()): Consider all actors that are on
    that node "colocated".
    If None: Consider the largest sub-set of actors that are all located on
    the same node (whatever that node is) as "colocated".

    Args:
        actors: The list of actor handles to split into "colocated" and
            "non colocated".
        node: The node defining "colocation" criterion. If provided, consider
            thos actors "colocated" that sit on this node. If None, use the
            largest subset within `actors` that are sitting on the same
            (any) node.

    Returns:
        Tuple of two lists: 1) Co-located ActorHandles, 2) non co-located
        ActorHandles.
    """
    if node == 'localhost':
        node = platform.node()
    hosts = ray.get([a.get_host.remote() for a in actors])
    if node is None:
        node_groups = defaultdict(set)
        for host, actor in zip(hosts, actors):
            node_groups[host].add(actor)
        max_ = -1
        largest_group = None
        for host in node_groups:
            if max_ < len(node_groups[host]):
                max_ = len(node_groups[host])
                largest_group = host
        non_co_located = []
        for host in node_groups:
            if host != largest_group:
                non_co_located.extend(list(node_groups[host]))
        return (list(node_groups[largest_group]), non_co_located)
    else:
        co_located = []
        non_co_located = []
        for host, a in zip(hosts, actors):
            if host == node:
                co_located.append(a)
            else:
                non_co_located.append(a)
        return (co_located, non_co_located)