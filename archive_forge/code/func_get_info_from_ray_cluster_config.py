import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple
import yaml
import ray  # noqa: F401
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler.tags import NODE_KIND_HEAD, NODE_KIND_WORKER, TAG_RAY_NODE_KIND
import psutil
def get_info_from_ray_cluster_config(cluster_config: str) -> Tuple[List[str], str, str, Optional[str], Optional[str]]:
    """Get information from Ray cluster config.

    Return list of host IPs, ssh user, ssh key file, and optional docker
    container.

    Args:
        cluster_config: Path to ray cluster config.

    Returns:
        Tuple of list of host IPs, ssh user name, ssh key file path,
            optional docker container name, optional cluster name.
    """
    from ray.autoscaler._private.commands import _bootstrap_config
    cli_logger.print(f'Retrieving cluster information from ray cluster file: {cluster_config}')
    cluster_config = os.path.expanduser(cluster_config)
    config = yaml.safe_load(open(cluster_config).read())
    config = _bootstrap_config(config, no_config_cache=True)
    provider = _get_node_provider(config['provider'], config['cluster_name'])
    head_nodes = provider.non_terminated_nodes({TAG_RAY_NODE_KIND: NODE_KIND_HEAD})
    worker_nodes = provider.non_terminated_nodes({TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
    hosts = [provider.external_ip(node) for node in head_nodes + worker_nodes]
    ssh_user = config['auth']['ssh_user']
    ssh_key = config['auth']['ssh_private_key']
    docker = None
    docker_config = config.get('docker', None)
    if docker_config:
        docker = docker_config.get('container_name', None)
    cluster_name = config.get('cluster_name', None)
    return (hosts, ssh_user, ssh_key, docker, cluster_name)