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
def new_port(lower_bound=10000, upper_bound=65535, denylist=None):
    if not denylist:
        denylist = set()
    port = random.randint(lower_bound, upper_bound)
    retry = 0
    while port in denylist:
        if retry > 100:
            break
        port = random.randint(lower_bound, upper_bound)
        retry += 1
    if retry > 100:
        raise ValueError(f'Failed to find a new port from the range {lower_bound}-{upper_bound}. Denylist: {denylist}')
    return port