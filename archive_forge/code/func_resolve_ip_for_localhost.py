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
def resolve_ip_for_localhost(address: str):
    """Convert to a remotely reachable IP if the address is "localhost"
            or "127.0.0.1". Otherwise do nothing.

    Args:
        address: This can be either a string containing a hostname (or an IP
            address) and a port or it can be just an IP address.

    Returns:
        The same address but with the local host replaced by remotely
            reachable IP.
    """
    if not address:
        raise ValueError(f'Malformed address: {address}')
    address_parts = address.split(':')
    if address_parts[0] == '127.0.0.1' or address_parts[0] == 'localhost':
        ip_address = get_node_ip_address()
        return ':'.join([ip_address] + address_parts[1:])
    else:
        return address