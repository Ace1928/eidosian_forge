import json
import logging
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional
from zlib import crc32
from ray._private.pydantic_compat import BaseModel
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.utils import DeploymentOptionUpdateType, get_random_letters
from ray.serve.generated.serve_pb2 import DeploymentVersion as DeploymentVersionProto
def requires_long_poll_broadcast(self, new_version):
    """Determines whether lightweightly updating an existing replica to the new
        version requires broadcasting through long poll that the running replicas has
        changed.
        """
    return self.deployment_config.max_concurrent_queries != new_version.deployment_config.max_concurrent_queries