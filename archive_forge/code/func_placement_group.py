import warnings
from typing import Dict, List, Optional, Union
import ray
from ray._private.auto_init_hook import auto_init_ray
from ray._private.client_mode_hook import client_mode_should_convert, client_mode_wrap
from ray._private.utils import hex_to_binary, get_ray_doc_version
from ray._raylet import PlacementGroupID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
@PublicAPI
@client_mode_wrap
def placement_group(bundles: List[Dict[str, float]], strategy: str='PACK', name: str='', lifetime: Optional[str]=None, _max_cpu_fraction_per_node: float=1.0) -> PlacementGroup:
    """Asynchronously creates a PlacementGroup.

    Args:
        bundles: A list of bundles which
            represent the resources requirements.
        strategy: The strategy to create the placement group.

         - "PACK": Packs Bundles into as few nodes as possible.
         - "SPREAD": Places Bundles across distinct nodes as even as possible.
         - "STRICT_PACK": Packs Bundles into one node. The group is
           not allowed to span multiple nodes.
         - "STRICT_SPREAD": Packs Bundles across distinct nodes.

        name: The name of the placement group.
        lifetime: Either `None`, which defaults to the placement group
            will fate share with its creator and will be deleted once its
            creator is dead, or "detached", which means the placement group
            will live as a global object independent of the creator.
        _max_cpu_fraction_per_node: (Experimental) Disallow placing bundles on nodes
            if it would cause the fraction of CPUs used by bundles from *any* placement
            group on the node to exceed this fraction. This effectively sets aside
            CPUs that placement groups cannot occupy on nodes. when
            `max_cpu_fraction_per_node < 1.0`, at least 1 CPU will be excluded from
            placement group scheduling. Note: This feature is experimental and is not
            recommended for use with autoscaling clusters (scale-up will not trigger
            properly).

    Raises:
        ValueError if bundle type is not a list.
        ValueError if empty bundle or empty resource bundles are given.
        ValueError if the wrong lifetime arguments are given.

    Return:
        PlacementGroup: Placement group object.
    """
    worker = ray._private.worker.global_worker
    worker.check_connected()
    if not isinstance(bundles, list):
        raise ValueError('The type of bundles must be list, got {}'.format(bundles))
    if not bundles:
        raise ValueError('The placement group `bundles` argument cannot contain an empty list')
    assert _max_cpu_fraction_per_node is not None
    if _max_cpu_fraction_per_node <= 0 or _max_cpu_fraction_per_node > 1:
        raise ValueError(f'Invalid argument `_max_cpu_fraction_per_node`: {_max_cpu_fraction_per_node}. _max_cpu_fraction_per_node must be a float between 0 and 1. ')
    for bundle in bundles:
        if len(bundle) == 0 or all((resource_value == 0 for resource_value in bundle.values())):
            raise ValueError(f'Bundles cannot be an empty dictionary or resources with only 0 values. Bundles: {bundles}')
        if 'object_store_memory' in bundle.keys():
            warnings.warn(f"Setting 'object_store_memory' for bundles is deprecated since it doesn't actually reserve the required object store memory. Use object spilling that's enabled by default (https://docs.ray.io/en/{get_ray_doc_version()}/ray-core/objects/object-spilling.html) instead to bypass the object store memory size limitation.", DeprecationWarning, stacklevel=1)
    if strategy not in VALID_PLACEMENT_GROUP_STRATEGIES:
        raise ValueError(f'Invalid placement group strategy {strategy}. Supported strategies are: {VALID_PLACEMENT_GROUP_STRATEGIES}.')
    if lifetime is None:
        detached = False
    elif lifetime == 'detached':
        detached = True
    else:
        raise ValueError("placement group `lifetime` argument must be either `None` or 'detached'")
    placement_group_id = worker.core_worker.create_placement_group(name, bundles, strategy, detached, _max_cpu_fraction_per_node)
    return PlacementGroup(placement_group_id)