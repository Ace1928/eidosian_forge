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
def node_ip_address_from_perspective(address: str):
    """IP address by which the local node can be reached *from* the `address`.

    Args:
        address: The IP address and port of any known live service on the
            network you care about.

    Returns:
        The IP address by which the local node can be reached from the address.
    """
    ip_address, port = address.split(':')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((ip_address, int(port)))
        node_ip_address = s.getsockname()[0]
    except OSError as e:
        node_ip_address = '127.0.0.1'
        if e.errno == errno.ENETUNREACH:
            try:
                host_name = socket.getfqdn(socket.gethostname())
                node_ip_address = socket.gethostbyname(host_name)
            except Exception:
                pass
    finally:
        s.close()
    return node_ip_address