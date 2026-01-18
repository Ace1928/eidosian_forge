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
def create_redis_client(redis_address, password=None):
    """Create a Redis client.

    Args:
        The IP address, port, and password of the Redis server.

    Returns:
        A Redis client.
    """
    import redis
    if not hasattr(create_redis_client, 'instances'):
        create_redis_client.instances = {}
    num_retries = ray_constants.START_REDIS_WAIT_RETRIES
    delay = 0.001
    for i in range(num_retries):
        cli = create_redis_client.instances.get(redis_address)
        if cli is None:
            redis_ip_address, redis_port = extract_ip_port(canonicalize_bootstrap_address_or_die(redis_address))
            cli = redis.StrictRedis(host=redis_ip_address, port=int(redis_port), password=password)
            create_redis_client.instances[redis_address] = cli
        try:
            cli.ping()
            return cli
        except Exception as e:
            create_redis_client.instances.pop(redis_address)
            if i >= num_retries - 1:
                raise RuntimeError(f'Unable to connect to Redis at {redis_address}: {e}')
            time.sleep(delay)
            delay = min(1, delay * 2)