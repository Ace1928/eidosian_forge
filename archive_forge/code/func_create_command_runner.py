import copy
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, Optional
from ray._private import ray_constants
from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler._private.gcp.node import GCPTPUNode
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
def create_command_runner(worker_id: int, accelerator_type: str, internal_ip: str, external_ip: str) -> CommandRunnerInterface:
    """Returns the correct base command runner."""
    common_args = {'internal_ip': internal_ip, 'external_ip': external_ip, 'worker_id': worker_id, 'accelerator_type': accelerator_type, 'log_prefix': '[tpu_worker_{}] '.format(worker_id) + log_prefix, 'node_id': node_id, 'provider': provider, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': use_internal_ip}
    if docker_config and docker_config['container_name'] != '':
        return TPUVMDockerCommandRunner(docker_config=docker_config, **common_args)
    else:
        return TPUVMSSHCommandRunner(**common_args)