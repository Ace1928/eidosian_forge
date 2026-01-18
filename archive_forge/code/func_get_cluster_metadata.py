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
def get_cluster_metadata(gcs_client) -> dict:
    """Get the cluster metadata from GCS.

    It is a blocking API.

    This will return None if `put_cluster_metadata` was never called.

    Params:
        gcs_client: The GCS client to perform KV operation GET.

    Returns:
        The cluster metadata in a dictinoary.

    Raises:
        RuntimeError if it fails to obtain cluster metadata from GCS.
    """
    return json.loads(gcs_client.internal_kv_get(usage_constant.CLUSTER_METADATA_KEY, namespace=ray_constants.KV_NAMESPACE_CLUSTER).decode('utf-8'))