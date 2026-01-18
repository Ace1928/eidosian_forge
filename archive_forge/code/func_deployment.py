import collections
import inspect
import logging
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from fastapi import APIRouter, FastAPI
import ray
from ray import cloudpickle
from ray._private.serialization import pickle_dumps
from ray.dag import DAGNode
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_graph_build import build as pipeline_build
from ray.serve._private.deployment_graph_build import (
from ray.serve._private.http_util import (
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.config import (
from ray.serve.context import (
from ray.serve.deployment import Application, Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.handle import DeploymentHandle
from ray.serve.multiplex import _ModelMultiplexWrapper
from ray.serve.schema import LoggingConfig, ServeInstanceDetails, ServeStatus
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.serve._private import api as _private_api  # isort:skip
@PublicAPI(stability='stable')
def deployment(_func_or_class: Optional[Callable]=None, name: Default[str]=DEFAULT.VALUE, version: Default[str]=DEFAULT.VALUE, num_replicas: Default[Optional[int]]=DEFAULT.VALUE, route_prefix: Default[Union[str, None]]=DEFAULT.VALUE, ray_actor_options: Default[Dict]=DEFAULT.VALUE, placement_group_bundles: Optional[List[Dict[str, float]]]=DEFAULT.VALUE, placement_group_strategy: Optional[str]=DEFAULT.VALUE, max_replicas_per_node: Default[int]=DEFAULT.VALUE, user_config: Default[Optional[Any]]=DEFAULT.VALUE, max_concurrent_queries: Default[int]=DEFAULT.VALUE, autoscaling_config: Default[Union[Dict, AutoscalingConfig, None]]=DEFAULT.VALUE, graceful_shutdown_wait_loop_s: Default[float]=DEFAULT.VALUE, graceful_shutdown_timeout_s: Default[float]=DEFAULT.VALUE, health_check_period_s: Default[float]=DEFAULT.VALUE, health_check_timeout_s: Default[float]=DEFAULT.VALUE, logging_config: Default[Union[Dict, LoggingConfig, None]]=DEFAULT.VALUE) -> Callable[[Callable], Deployment]:
    """Decorator that converts a Python class to a `Deployment`.

    Example:

    .. code-block:: python

        from ray import serve

        @serve.deployment(num_replicas=2)
        class MyDeployment:
            pass

        app = MyDeployment.bind()

    Args:
        name: Name uniquely identifying this deployment within the application.
            If not provided, the name of the class or function is used.
        num_replicas: Number of replicas to run that handle requests to
            this deployment. Defaults to 1.
        autoscaling_config: Parameters to configure autoscaling behavior. If this
            is set, `num_replicas` cannot be set.
        route_prefix: [DEPRECATED] Route prefix should be set per-application
            through `serve.run()`.
        ray_actor_options: Options to pass to the Ray Actor decorator, such as
            resource requirements. Valid options are: `accelerator_type`, `memory`,
            `num_cpus`, `num_gpus`, `object_store_memory`, `resources`,
            and `runtime_env`.
        placement_group_bundles: Defines a set of placement group bundles to be
            scheduled *for each replica* of this deployment. The replica actor will
            be scheduled in the first bundle provided, so the resources specified in
            `ray_actor_options` must be a subset of the first bundle's resources. All
            actors and tasks created by the replica actor will be scheduled in the
            placement group by default (`placement_group_capture_child_tasks` is set
            to True).
        placement_group_strategy: Strategy to use for the replica placement group
            specified via `placement_group_bundles`. Defaults to `PACK`.
        user_config: Config to pass to the reconfigure method of the deployment. This
            can be updated dynamically without restarting the replicas of the
            deployment. The user_config must be fully JSON-serializable.
        max_concurrent_queries: Maximum number of queries that are sent to a
            replica of this deployment without receiving a response. Defaults to 100.
        health_check_period_s: Duration between health check calls for the replica.
            Defaults to 10s. The health check is by default a no-op Actor call to the
            replica, but you can define your own health check using the "check_health"
            method in your deployment that raises an exception when unhealthy.
        health_check_timeout_s: Duration in seconds, that replicas wait for a health
            check method to return before considering it as failed. Defaults to 30s.
        graceful_shutdown_wait_loop_s: Duration that replicas wait until there is
            no more work to be done before shutting down. Defaults to 2s.
        graceful_shutdown_timeout_s: Duration to wait for a replica to gracefully
            shut down before being forcefully killed. Defaults to 20s.
        max_replicas_per_node: [EXPERIMENTAL] The max number of deployment replicas can
            run on a single node. Valid values are None (no limitation)
            or an integer in the range of [1, 100].
            Defaults to no limitation.

    Returns:
        `Deployment`
    """
    user_configured_option_names = [option for option, value in locals().items() if option != '_func_or_class' and value is not DEFAULT.VALUE]
    if num_replicas == 0:
        raise ValueError('num_replicas is expected to larger than 0')
    if num_replicas not in [DEFAULT.VALUE, None] and autoscaling_config not in [DEFAULT.VALUE, None]:
        raise ValueError('Manually setting num_replicas is not allowed when autoscaling_config is provided.')
    if version is not DEFAULT.VALUE:
        logger.warning('DeprecationWarning: `version` in `@serve.deployment` has been deprecated. Explicitly specifying version will raise an error in the future!')
    if route_prefix is not DEFAULT.VALUE:
        logger.warning('DeprecationWarning: `route_prefix` in `@serve.deployment` has been deprecated. To specify a route prefix for an application, pass it into `serve.run` instead.')
    if isinstance(logging_config, LoggingConfig):
        logging_config = logging_config.dict()
    deployment_config = DeploymentConfig.from_default(num_replicas=num_replicas if num_replicas is not None else 1, user_config=user_config, max_concurrent_queries=max_concurrent_queries, autoscaling_config=autoscaling_config, graceful_shutdown_wait_loop_s=graceful_shutdown_wait_loop_s, graceful_shutdown_timeout_s=graceful_shutdown_timeout_s, health_check_period_s=health_check_period_s, health_check_timeout_s=health_check_timeout_s, logging_config=logging_config)
    deployment_config.user_configured_option_names = set(user_configured_option_names)

    def decorator(_func_or_class):
        replica_config = ReplicaConfig.create(_func_or_class, init_args=None, init_kwargs=None, ray_actor_options=ray_actor_options if ray_actor_options is not DEFAULT.VALUE else None, placement_group_bundles=placement_group_bundles if placement_group_bundles is not DEFAULT.VALUE else None, placement_group_strategy=placement_group_strategy if placement_group_strategy is not DEFAULT.VALUE else None, max_replicas_per_node=max_replicas_per_node if max_replicas_per_node is not DEFAULT.VALUE else None)
        return Deployment(name if name is not DEFAULT.VALUE else _func_or_class.__name__, deployment_config, replica_config, version=version if version is not DEFAULT.VALUE else None, route_prefix=route_prefix, _internal=True)
    return decorator(_func_or_class) if callable(_func_or_class) else decorator