from collections import defaultdict, deque
import logging
import platform
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
import ray
from ray.actor import ActorClass, ActorHandle
def create_colocated_actors(actor_specs: Sequence[Tuple[Type, Any, Any, int]], node: Optional[str]='localhost', max_attempts: int=10) -> Dict[Type, List[ActorHandle]]:
    """Create co-located actors of any type(s) on any node.

    Args:
        actor_specs: Tuple/list with tuples consisting of: 1) The
            (already @ray.remote) class(es) to construct, 2) c'tor args,
            3) c'tor kwargs, and 4) the number of actors of that class with
            given args/kwargs to construct.
        node: The node to co-locate the actors on. By default ("localhost"),
            place the actors on the node the caller of this function is
            located on. Use None for indicating that any (resource fulfilling)
            node in the cluster may be used.
        max_attempts: The maximum number of co-location attempts to
            perform before throwing an error.

    Returns:
        A dict mapping the created types to the list of n ActorHandles
        created (and co-located) for that type.
    """
    if node == 'localhost':
        node = platform.node()
    ok = [[] for _ in range(len(actor_specs))]
    for attempt in range(max_attempts):
        all_good = True
        for i, (typ, args, kwargs, count) in enumerate(actor_specs):
            args = args or []
            kwargs = kwargs or {}
            if len(ok[i]) < count:
                co_located = try_create_colocated(cls=typ, args=args, kwargs=kwargs, count=count * (attempt + 1), node=node)
                if node is None:
                    node = ray.get(co_located[0].get_host.remote())
                ok[i].extend(co_located)
                if len(ok[i]) < count:
                    all_good = False
            if len(ok[i]) > count:
                for a in ok[i][count:]:
                    a.__ray_terminate__.remote()
                ok[i] = ok[i][:count]
        if all_good:
            return ok
    raise Exception('Unable to create enough colocated actors -> aborting.')