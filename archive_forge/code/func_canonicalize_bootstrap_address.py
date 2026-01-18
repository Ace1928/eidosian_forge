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
def canonicalize_bootstrap_address(addr: str, temp_dir: Optional[str]=None) -> Optional[str]:
    """Canonicalizes Ray cluster bootstrap address to host:port.
    Reads address from the environment if needed.

    This function should be used to process user supplied Ray cluster address,
    via ray.init() or `--address` flags, before using the address to connect.

    Returns:
        Ray cluster address string in <host:port> format or None if the caller
        should start a local Ray instance.
    """
    if addr is None or addr == 'auto':
        addr = get_ray_address_from_environment(addr, temp_dir)
    if addr is None or addr == 'local':
        return None
    try:
        bootstrap_address = resolve_ip_for_localhost(addr)
    except Exception:
        logger.exception(f'Failed to convert {addr} to host:port')
        raise
    return bootstrap_address