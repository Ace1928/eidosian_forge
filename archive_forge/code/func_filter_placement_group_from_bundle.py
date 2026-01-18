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
def filter_placement_group_from_bundle(bundle: ResourceBundle):
    """filter placement group from bundle resource name. returns
        filtered bundle and a bool indicate if the bundle is using
        placement group.

        Example: {"CPU_group_groupid": 1} returns {"CPU": 1}, True
                 {"memory": 1} return {"memory": 1}, False
        """
    using_placement_group = False
    result_bundle = dict()
    for pg_resource_str, resource_count in bundle.items():
        resource_name, pg_name, _ = parse_placement_group_resource_str(pg_resource_str)
        result_bundle[resource_name] = resource_count
        if pg_name:
            using_placement_group = True
    return (result_bundle, using_placement_group)