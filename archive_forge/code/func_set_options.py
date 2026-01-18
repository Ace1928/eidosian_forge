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
@Deprecated(message='This was intended for use with the `serve.build` Python API (which has been deprecated). Use `.options()` instead.')
def set_options(self, func_or_class: Optional[Callable]=None, name: Default[str]=DEFAULT.VALUE, version: Default[str]=DEFAULT.VALUE, num_replicas: Default[Optional[int]]=DEFAULT.VALUE, route_prefix: Default[Union[str, None]]=DEFAULT.VALUE, ray_actor_options: Default[Optional[Dict]]=DEFAULT.VALUE, user_config: Default[Optional[Any]]=DEFAULT.VALUE, max_concurrent_queries: Default[int]=DEFAULT.VALUE, autoscaling_config: Default[Union[Dict, AutoscalingConfig, None]]=DEFAULT.VALUE, graceful_shutdown_wait_loop_s: Default[float]=DEFAULT.VALUE, graceful_shutdown_timeout_s: Default[float]=DEFAULT.VALUE, health_check_period_s: Default[float]=DEFAULT.VALUE, health_check_timeout_s: Default[float]=DEFAULT.VALUE, _internal: bool=False) -> None:
    """Overwrite this deployment's options in-place.

        Only those options passed in will be updated, all others will remain
        unchanged.

        Refer to the @serve.deployment decorator docstring for all non-private
        arguments.
        """
    if not _internal:
        warnings.warn('`.set_options()` is deprecated. Use `.options()` or an application builder function instead.')
    validated = self.options(func_or_class=func_or_class, name=name, version=version, route_prefix=route_prefix, num_replicas=num_replicas, ray_actor_options=ray_actor_options, user_config=user_config, max_concurrent_queries=max_concurrent_queries, autoscaling_config=autoscaling_config, graceful_shutdown_wait_loop_s=graceful_shutdown_wait_loop_s, graceful_shutdown_timeout_s=graceful_shutdown_timeout_s, health_check_period_s=health_check_period_s, health_check_timeout_s=health_check_timeout_s, _internal=_internal)
    self._name = validated._name
    self._version = validated._version
    self._route_prefix = validated._route_prefix
    self._deployment_config = validated._deployment_config
    self._replica_config = validated._replica_config