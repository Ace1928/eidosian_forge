import os
import sys
import warnings
from typing import Optional
import psutil
import ray
from packaging import version
from modin.config import (
from modin.core.execution.utils import set_env
from modin.error_message import ErrorMessage
from .engine_wrapper import ObjectRefTypes, RayWrapper
def initialize_ray(override_is_cluster=False, override_redis_address: str=None, override_redis_password: str=None):
    """
    Initialize Ray based on parameters, ``modin.config`` variables and internal defaults.

    Parameters
    ----------
    override_is_cluster : bool, default: False
        Whether to override the detection of Modin being run in a cluster
        and always assume this runs on cluster head node.
        This also overrides Ray worker detection and always runs the initialization
        function (runs from main thread only by default).
        If not specified, ``modin.config.IsRayCluster`` variable is used.
    override_redis_address : str, optional
        What Redis address to connect to when running in Ray cluster.
        If not specified, ``modin.config.RayRedisAddress`` is used.
    override_redis_password : str, optional
        What password to use when connecting to Redis.
        If not specified, ``modin.config.RayRedisPassword`` is used.
    """
    env_vars = {'__MODIN_AUTOIMPORT_PANDAS__': '1', 'PYTHONWARNINGS': 'ignore::FutureWarning'}
    if GithubCI.get():
        env_vars.update({'AWS_ACCESS_KEY_ID': CIAWSAccessKeyID.get(), 'AWS_SECRET_ACCESS_KEY': CIAWSSecretAccessKey.get()})
    extra_init_kw = {}
    is_cluster = override_is_cluster or IsRayCluster.get()
    if not ray.is_initialized() or override_is_cluster:
        redis_address = override_redis_address or RayRedisAddress.get()
        redis_password = (ray.ray_constants.REDIS_DEFAULT_PASSWORD if is_cluster else RayRedisPassword.get()) if override_redis_password is None and RayRedisPassword.get_value_source() == ValueSource.DEFAULT else override_redis_password or RayRedisPassword.get()
        if is_cluster:
            extra_init_kw['runtime_env'] = {'env_vars': env_vars}
            ray.init(address=redis_address or 'auto', include_dashboard=False, ignore_reinit_error=True, _redis_password=redis_password, **extra_init_kw)
        else:
            object_store_memory = _get_object_store_memory()
            ray_init_kwargs = {'num_cpus': CpuCount.get(), 'num_gpus': GpuCount.get(), 'include_dashboard': False, 'ignore_reinit_error': True, 'object_store_memory': object_store_memory, '_redis_password': redis_password, '_memory': object_store_memory, **extra_init_kw}
            with set_env(**env_vars):
                ray.init(**ray_init_kwargs)
        if StorageFormat.get() == 'Cudf':
            from modin.core.execution.ray.implementations.cudf_on_ray.partitioning import GPU_MANAGERS, GPUManager
            if not GPU_MANAGERS:
                for i in range(GpuCount.get()):
                    GPU_MANAGERS.append(GPUManager.remote(i))
    runtime_env_vars = ray.get_runtime_context().runtime_env.get('env_vars', {})
    for varname, varvalue in env_vars.items():
        if str(runtime_env_vars.get(varname, '')) != str(varvalue):
            if is_cluster:
                ErrorMessage.single_warning('When using a pre-initialized Ray cluster, please ensure that the runtime env ' + f'sets environment variable {varname} to {varvalue}')
    num_cpus = int(ray.cluster_resources()['CPU'])
    num_gpus = int(ray.cluster_resources().get('GPU', 0))
    if StorageFormat.get() == 'Cudf':
        NPartitions._put(num_gpus)
    else:
        NPartitions._put(num_cpus)
    if _RAY_IGNORE_UNHANDLED_ERRORS_VAR not in os.environ:
        os.environ[_RAY_IGNORE_UNHANDLED_ERRORS_VAR] = '1'