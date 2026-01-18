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
def extract_ip_port(bootstrap_address: str):
    if ':' not in bootstrap_address:
        raise ValueError(f"Malformed address {bootstrap_address}. Expected '<host>:<port>'.")
    ip, _, port = bootstrap_address.rpartition(':')
    try:
        port = int(port)
    except ValueError:
        raise ValueError(f'Malformed address port {port}. Must be an integer.')
    if port < 1024 or port > 65535:
        raise ValueError(f'Invalid address port {port}. Must be between 1024 and 65535 (inclusive).')
    return (ip, port)