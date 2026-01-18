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
def find_bootstrap_address(temp_dir: Optional[str]):
    """Finds the latest Ray cluster address to connect to, if any. This is the
    GCS address connected to by the last successful `ray start`."""
    return ray._private.utils.read_ray_address(temp_dir)