import collections
import copy
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from numbers import Number, Real
from typing import Any, Dict, List, Optional, Tuple, Union
import ray
import ray._private.services as services
from ray._private.utils import (
from ray.autoscaler._private import constants
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.docker import validate_docker_config
from ray.autoscaler._private.local.config import prepare_local
from ray.autoscaler._private.providers import _get_default_config
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def merge_legacy_yaml_with_defaults(merged_config: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite legacy config's available node types after it has been merged
    with defaults yaml.
    """
    cli_logger.warning('Converting legacy cluster config to a multi node type cluster config. Multi-node-type cluster configs are the recommended format for configuring Ray clusters. See the docs for more information:\nhttps://docs.ray.io/en/master/cluster/config.html#full-configuration')
    default_head_type = merged_config['head_node_type']
    assert len(merged_config['available_node_types'].keys()) == 2
    default_worker_type = (merged_config['available_node_types'].keys() - {default_head_type}).pop()
    if merged_config['head_node']:
        head_node_info = {'node_config': merged_config['head_node'], 'resources': merged_config['head_node'].get('resources') or {}, 'min_workers': 0, 'max_workers': 0}
    else:
        head_node_info = merged_config['available_node_types'][default_head_type]
    if merged_config['worker_nodes']:
        worker_node_info = {'node_config': merged_config['worker_nodes'], 'resources': merged_config['worker_nodes'].get('resources') or {}, 'min_workers': merged_config.get('min_workers', 0), 'max_workers': merged_config['max_workers']}
    else:
        worker_node_info = merged_config['available_node_types'][default_worker_type]
    merged_config['available_node_types'] = {NODE_TYPE_LEGACY_HEAD: head_node_info, NODE_TYPE_LEGACY_WORKER: worker_node_info}
    merged_config['head_node_type'] = NODE_TYPE_LEGACY_HEAD
    merged_config['head_node'].pop('resources', None)
    merged_config['worker_nodes'].pop('resources', None)
    return merged_config