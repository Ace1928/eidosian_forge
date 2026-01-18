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
def get_shared_memory_bytes():
    """Get the size of the shared memory file system.

    Returns:
        The size of the shared memory file system in bytes.
    """
    assert sys.platform == 'linux' or sys.platform == 'linux2'
    shm_fd = os.open('/dev/shm', os.O_RDONLY)
    try:
        shm_fs_stats = os.fstatvfs(shm_fd)
        shm_avail = shm_fs_stats.f_bsize * shm_fs_stats.f_bavail
    finally:
        os.close(shm_fd)
    return shm_avail