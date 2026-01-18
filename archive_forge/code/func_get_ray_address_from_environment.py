import base64
import collections
import errno
import io
import json
import logging
import mmap
import multiprocessing
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, IO, AnyStr
import psutil
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
from ray._raylet import GcsClient, GcsClientOptions
from ray.core.generated.common_pb2 import Language
from ray._private.ray_constants import RAY_NODE_IP_FILENAME
def get_ray_address_from_environment(addr: str, temp_dir: Optional[str]):
    """Attempts to find the address of Ray cluster to use, in this order:

    1. Use RAY_ADDRESS if defined and nonempty.
    2. If no address is provided or the provided address is "auto", use the
    address in /tmp/ray/ray_current_cluster if available. This will error if
    the specified address is None and there is no address found. For "auto",
    we will fallback to connecting to any detected Ray cluster (legacy).
    3. Otherwise, use the provided address.

    Returns:
        A string to pass into `ray.init(address=...)`, e.g. ip:port, `auto`.
    """
    env_addr = os.environ.get(ray_constants.RAY_ADDRESS_ENVIRONMENT_VARIABLE)
    if env_addr is not None and env_addr != '':
        addr = env_addr
    if addr is not None and addr != 'auto':
        return addr
    gcs_addrs = find_gcs_addresses()
    bootstrap_addr = find_bootstrap_address(temp_dir)
    if len(gcs_addrs) > 1 and bootstrap_addr is not None:
        logger.warning(f'Found multiple active Ray instances: {gcs_addrs}. Connecting to latest cluster at {bootstrap_addr}. You can override this by setting the `--address` flag or `RAY_ADDRESS` environment variable.')
    elif len(gcs_addrs) > 0 and addr == 'auto':
        bootstrap_addr = list(gcs_addrs).pop()
    if bootstrap_addr is None:
        if addr is None:
            return None
        else:
            raise ConnectionError('Could not find any running Ray instance. Please specify the one to connect to by setting `--address` flag or `RAY_ADDRESS` environment variable.')
    return bootstrap_addr