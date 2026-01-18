import copy
import hashlib
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Union
import botocore
from ray.autoscaler._private.aws.utils import client_cache, resource_cache
from ray.autoscaler.tags import NODE_KIND_HEAD, TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_KIND
@staticmethod
def resolve_instance_profile_name(config: Dict[str, Any], default_instance_profile_name: str) -> str:
    """Get default cloudwatch instance profile name.

        Args:
            config: provider section of cluster config file.
            default_instance_profile_name: default ray instance profile name.

        Returns:
            default cloudwatch instance profile name if cloudwatch config file
                exists.
            default ray instance profile name if cloudwatch config file
                doesn't exist.
        """
    cwa_cfg_exists = CloudwatchHelper.cloudwatch_config_exists(config, CloudwatchConfigType.AGENT.value)
    return CLOUDWATCH_RAY_INSTANCE_PROFILE if cwa_cfg_exists else default_instance_profile_name