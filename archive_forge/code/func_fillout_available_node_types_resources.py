import copy
import logging
import time
from functools import wraps
from threading import RLock
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import googleapiclient
from ray.autoscaler._private.gcp.config import (
from ray.autoscaler._private.gcp.node import GCPTPU  # noqa
from ray.autoscaler._private.gcp.node import (
from ray.autoscaler._private.gcp.tpu_command_runner import TPUCommandRunner
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
@staticmethod
def fillout_available_node_types_resources(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill out TPU resources to the cluster config.

        To enable TPU pod autoscaling, we provide the TPU accelerator
        type as a resource that only exists on worker 0 of the pod slice.
        For instance, a v4-16 should have the resource labels:
            worker 0: resources = {"TPU": 4, "TPU-v4-16-head": 1}
            worker 1: resources = {"TPU": 4}

        For the autoscaler to correctly process the demands of
        creating a new TPU pod, then the autoscaler must know what
        a TPU pod is in the form of the TPU accelerator resource.

        Therefore we fill out TPU pods appropriately by providing the
        expected resource which we can deduce from the cluster config.

        """
    if 'available_node_types' not in cluster_config:
        return cluster_config
    cluster_config = copy.deepcopy(cluster_config)
    available_node_types = cluster_config['available_node_types']
    for node_type in available_node_types:
        node_config = available_node_types[node_type]['node_config']
        if get_node_type(node_config) == GCPNodeType.TPU:
            autodetected_resources = {}
            accelerator_type = ''
            if 'acceleratorType' in node_config:
                accelerator_type = node_config['acceleratorType']
            elif 'acceleratorConfig' in node_config:
                accelerator_type = tpu_accelerator_config_to_type(node_config['acceleratorConfig'])
            if not accelerator_type:
                continue
            autodetected_resources[f'TPU-{accelerator_type}-head'] = 1
            available_node_types[node_type]['resources'].update(autodetected_resources)
    return cluster_config