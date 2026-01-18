import copy
import json
import logging
import os
import subprocess
import sys
import time
from threading import RLock
from types import ModuleType
from typing import Any, Dict, Optional
import yaml
import ray
import ray._private.ray_constants as ray_constants
from ray.autoscaler._private.fake_multi_node.command_runner import (
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def create_node_spec(head: bool, docker_image: str, mounted_cluster_dir: str, mounted_node_dir: str, num_cpus: int=2, num_gpus: int=0, resources: Optional[Dict]=None, env_vars: Optional[Dict]=None, host_gcs_port: int=16379, host_object_manager_port: int=18076, host_client_port: int=10002, volume_dir: Optional[str]=None, node_state_path: Optional[str]=None, docker_status_path: Optional[str]=None, docker_compose_path: Optional[str]=None, bootstrap_config_path: Optional[str]=None, private_key_path: Optional[str]=None, public_key_path: Optional[str]=None):
    node_spec = copy.deepcopy(DOCKER_NODE_SKELETON)
    node_spec['image'] = docker_image
    bootstrap_cfg_path_on_container = '/home/ray/ray_bootstrap_config.yaml'
    bootstrap_key_path_on_container = '/home/ray/ray_bootstrap_key.pem'
    resources = resources or {}
    ensure_ssh = '((sudo apt update && sudo apt install -y openssh-server && sudo service ssh start) || true)' if not bool(int(os.environ.get('RAY_HAS_SSH', '0') or '0')) else 'sudo service ssh start'
    cmd_kwargs = dict(ensure_ssh=ensure_ssh, num_cpus=num_cpus, num_gpus=num_gpus, resources=json.dumps(resources, indent=None), volume_dir=volume_dir, autoscaling_config=bootstrap_cfg_path_on_container)
    env_vars = env_vars or {}
    fake_cluster_dev_dir = os.environ.get('FAKE_CLUSTER_DEV', '')
    if fake_cluster_dev_dir:
        if fake_cluster_dev_dir == 'auto':
            local_ray_dir = os.path.dirname(ray.__file__)
        else:
            local_ray_dir = fake_cluster_dev_dir
        os.environ['FAKE_CLUSTER_DEV'] = local_ray_dir
        mj = sys.version_info.major
        mi = sys.version_info.minor
        fake_modules_str = os.environ.get('FAKE_CLUSTER_DEV_MODULES', 'autoscaler')
        fake_modules = fake_modules_str.split(',')
        docker_ray_dir = f'/home/ray/anaconda3/lib/python{mj}.{mi}/site-packages/ray'
        node_spec['volumes'] += [f'{local_ray_dir}/{module}:{docker_ray_dir}/{module}:ro' for module in fake_modules]
        env_vars['FAKE_CLUSTER_DEV'] = local_ray_dir
        env_vars['FAKE_CLUSTER_DEV_MODULES'] = fake_modules_str
        os.environ['FAKE_CLUSTER_DEV_MODULES'] = fake_modules_str
    if head:
        node_spec['command'] = DOCKER_HEAD_CMD.format(**cmd_kwargs)
        node_spec['ports'] = [f'{host_gcs_port}:{ray_constants.DEFAULT_PORT}', f'{host_object_manager_port}:8076', f'{host_client_port}:10001']
        node_spec['volumes'] += [f'{host_dir(node_state_path)}:{node_state_path}', f'{host_dir(docker_status_path)}:{docker_status_path}', f'{host_dir(docker_compose_path)}:{docker_compose_path}', f'{host_dir(bootstrap_config_path)}:{bootstrap_cfg_path_on_container}', f'{host_dir(private_key_path)}:{bootstrap_key_path_on_container}']
        for filename in [node_state_path, docker_status_path, bootstrap_config_path]:
            if not os.path.exists(filename):
                with open(filename, 'wt') as f:
                    f.write('{}')
    else:
        node_spec['command'] = DOCKER_WORKER_CMD.format(**cmd_kwargs)
        node_spec['depends_on'] = [FAKE_HEAD_NODE_ID]
    node_spec['volumes'] += [f'{host_dir(mounted_cluster_dir)}:/cluster/shared', f'{host_dir(mounted_node_dir)}:/cluster/node', f'{host_dir(public_key_path)}:/home/ray/.ssh/authorized_keys']
    env_vars.setdefault('RAY_HAS_SSH', os.environ.get('RAY_HAS_SSH', ''))
    env_vars.setdefault('RAY_TEMPDIR', os.environ.get('RAY_TEMPDIR', ''))
    env_vars.setdefault('RAY_HOSTDIR', os.environ.get('RAY_HOSTDIR', ''))
    node_spec['environment'] = [f'{k}={v}' for k, v in env_vars.items()]
    return node_spec