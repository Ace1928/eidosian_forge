import inspect
import logging
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray.dag.class_node import ClassNode
from ray.dag.dag_node import DAGNodeBase
from ray.dag.function_node import FunctionNode
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import DEFAULT, Default
from ray.serve.config import AutoscalingConfig
from ray.serve.context import _get_global_client
from ray.serve.handle import RayServeHandle, RayServeSyncHandle
from ray.serve.schema import DeploymentSchema, LoggingConfig, RayActorOptionsSchema
from ray.util.annotations import Deprecated, PublicAPI
def deployment_to_schema(d: Deployment, include_route_prefix: bool=True) -> DeploymentSchema:
    """Converts a live deployment object to a corresponding structured schema.

    Args:
        d: Deployment object to convert
        include_route_prefix: Whether to include the route_prefix in the returned
            schema. This should be set to False if the schema will be included in a
            higher-level object describing an application, and you want to place
            route_prefix at the application level.
    """
    if d.ray_actor_options is not None:
        ray_actor_options_schema = RayActorOptionsSchema.parse_obj(d.ray_actor_options)
    else:
        ray_actor_options_schema = None
    deployment_options = {'name': d.name, 'num_replicas': None if d._deployment_config.autoscaling_config else d.num_replicas, 'max_concurrent_queries': d.max_concurrent_queries, 'user_config': d.user_config, 'autoscaling_config': d._deployment_config.autoscaling_config, 'graceful_shutdown_wait_loop_s': d._deployment_config.graceful_shutdown_wait_loop_s, 'graceful_shutdown_timeout_s': d._deployment_config.graceful_shutdown_timeout_s, 'health_check_period_s': d._deployment_config.health_check_period_s, 'health_check_timeout_s': d._deployment_config.health_check_timeout_s, 'ray_actor_options': ray_actor_options_schema, 'placement_group_strategy': d._replica_config.placement_group_strategy, 'placement_group_bundles': d._replica_config.placement_group_bundles, 'max_replicas_per_node': d._replica_config.max_replicas_per_node, 'logging_config': d._deployment_config.logging_config}
    if include_route_prefix:
        deployment_options['route_prefix'] = d.route_prefix
    for option in list(deployment_options.keys()):
        if option != 'name' and option not in d._deployment_config.user_configured_option_names:
            del deployment_options[option]
    return DeploymentSchema(**deployment_options)