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
def fillout_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _get_default_config(config['provider'])
    defaults.update(config)
    merged_config = copy.deepcopy(defaults)
    merged_config['auth'] = merged_config.get('auth', {})
    is_legacy_config = 'available_node_types' not in config and ('head_node' in config or 'worker_nodes' in config)
    if is_legacy_config:
        merged_config = merge_legacy_yaml_with_defaults(merged_config)
    merged_config.pop('min_workers', None)
    translate_trivial_legacy_config(merged_config)
    return merged_config