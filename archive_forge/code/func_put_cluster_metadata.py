import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def put_cluster_metadata(gcs_client) -> None:
    """Generate the cluster metadata and store it to GCS.

    It is a blocking API.

    Params:
        gcs_client: The GCS client to perform KV operation PUT.

    Raises:
        gRPC exceptions if PUT fails.
    """
    metadata = _generate_cluster_metadata()
    gcs_client.internal_kv_put(usage_constant.CLUSTER_METADATA_KEY, json.dumps(metadata).encode(), overwrite=True, namespace=ray_constants.KV_NAMESPACE_CLUSTER)
    return metadata