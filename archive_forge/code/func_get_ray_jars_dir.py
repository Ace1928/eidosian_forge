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
def get_ray_jars_dir():
    """Return a directory where all ray-related jars and
    their dependencies locate."""
    current_dir = RAY_PATH
    jars_dir = os.path.abspath(os.path.join(current_dir, 'jars'))
    if not os.path.exists(jars_dir):
        raise RuntimeError('Ray jars is not packaged into ray. Please build ray with java enabled (set env var RAY_INSTALL_JAVA=1)')
    return os.path.abspath(os.path.join(current_dir, 'jars'))