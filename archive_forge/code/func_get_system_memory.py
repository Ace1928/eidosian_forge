import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def get_system_memory(memory_limit_filename='/sys/fs/cgroup/memory/memory.limit_in_bytes', memory_limit_filename_v2='/sys/fs/cgroup/memory.max'):
    """Return the total amount of system memory in bytes.

    Returns:
        The total amount of system memory in bytes.
    """
    docker_limit = None
    if os.path.exists(memory_limit_filename):
        with open(memory_limit_filename, 'r') as f:
            docker_limit = int(f.read().strip())
    elif os.path.exists(memory_limit_filename_v2):
        with open(memory_limit_filename_v2, 'r') as f:
            max_file = f.read().strip()
            if max_file.isnumeric():
                docker_limit = int(max_file)
            else:
                docker_limit = None
    psutil_memory_in_bytes = psutil.virtual_memory().total
    if docker_limit is not None:
        return min(docker_limit, psutil_memory_in_bytes)
    return psutil_memory_in_bytes