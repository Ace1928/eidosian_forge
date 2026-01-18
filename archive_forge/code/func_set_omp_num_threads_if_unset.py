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
def set_omp_num_threads_if_unset() -> bool:
    """Set the OMP_NUM_THREADS to default to num cpus assigned to the worker

    This function sets the environment variable OMP_NUM_THREADS for the worker,
    if the env is not previously set and it's running in worker (WORKER_MODE).

    Returns True if OMP_NUM_THREADS is set in this function.

    """
    num_threads_from_env = os.environ.get('OMP_NUM_THREADS')
    if num_threads_from_env is not None:
        return False
    runtime_ctx = ray.get_runtime_context()
    if runtime_ctx.worker.mode != ray._private.worker.WORKER_MODE:
        return False
    num_assigned_cpus = runtime_ctx.get_assigned_resources().get('CPU')
    if num_assigned_cpus is None:
        logger.debug('[ray] Forcing OMP_NUM_THREADS=1 to avoid performance degradation with many workers (issue #6998). You can override this by explicitly setting OMP_NUM_THREADS, or changing num_cpus.')
        num_assigned_cpus = 1
    import math
    omp_num_threads = max(math.floor(num_assigned_cpus), 1)
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    return True