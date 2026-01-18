import copy
import itertools
import json
import logging
import os
import time
from collections import Counter
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Set, Tuple
import boto3
import botocore
from packaging.version import Version
from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import (
from ray.autoscaler._private.aws.utils import (
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.providers import _PROVIDER_PRETTY_NAMES
from ray.autoscaler._private.util import check_legacy_fields
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def print_info(resource_string: str, key: str, src_key: str, allowed_tags: Optional[List[str]]=None, list_value: bool=False) -> None:
    if allowed_tags is None:
        allowed_tags = ['default']
    node_tags = {}
    unique_settings = set()
    for node_type_key, node_type in config['available_node_types'].items():
        node_tags[node_type_key] = {}
        tag = _log_info[src_key][node_type_key]
        if tag in allowed_tags:
            node_tags[node_type_key][tag] = True
        setting = node_type['node_config'].get(key)
        if list_value:
            unique_settings.add(tuple(setting))
        else:
            unique_settings.add(setting)
    head_value_str = head_node_config[key]
    if list_value:
        head_value_str = cli_logger.render_list(head_value_str)
    if len(unique_settings) == 1:
        cli_logger.labeled_value(resource_string + ' (all available node types)', '{}', head_value_str, _tags=node_tags[config['head_node_type']])
    else:
        cli_logger.labeled_value(resource_string + f' ({head_node_type})', '{}', head_value_str, _tags=node_tags[head_node_type])
        for node_type_key, node_type in config['available_node_types'].items():
            if node_type_key == head_node_type:
                continue
            workers_value_str = node_type['node_config'][key]
            if list_value:
                workers_value_str = cli_logger.render_list(workers_value_str)
            cli_logger.labeled_value(resource_string + f' ({node_type_key})', '{}', workers_value_str, _tags=node_tags[node_type_key])